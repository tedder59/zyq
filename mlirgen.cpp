// Cambricon is pleased to support the open source community by making zyq available.
//
// Copyright (C) [2020-2023] by Cambricon Inc.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "mlirgen.h"
#include "ast.h"
#include "dialect.h"
#include <numeric>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::zyq;
using namespace zyq;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

class MLIRGenImpl
{
public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
    
    mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (FunctionAST &F : moduleAST)
        {
            auto func = mlirGen(F);
            if (!func) return nullptr;
            theModule.push_back(func);
        }
        
        if (failed(mlir::verify(theModule)))
        {
            theModule.emitError("module verification error");
            return nullptr;
        }
        
        return theModule;
    }

private:
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;
    llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
    
    mlir::Location loc(Location loc) {
        return builder.getFileLineColLoc(
            builder.getIdentifier(*loc.file), loc.line, loc.col);
    }
    
    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (symbolTable.count(var)) return mlir::failure();
        symbolTable.insert(var, value);
        return mlir::success();
    }

    mlir::FuncOp mlirGen(PrototypeAST &proto) {
        auto location = loc(proto.loc());
        llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                                   getType(VarType{}));
        auto func_type = builder.getFunctionType(arg_types, llvm::None);
        return mlir::FuncOp::create(location, proto.getName(), func_type);
    }

    mlir::FuncOp mlirGen(FunctionAST &funcAST) {
        ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

        mlir::FuncOp function(mlirGen(*funcAST.getProto()));
        if (!function) return nullptr;

        auto &entryBlock = *function.addEntryBlock();
        auto protoArgs = funcAST.getProto()->getArgs();

        for (const auto &name_value : llvm::zip(protoArgs, entryBlock.getArguments()))
        {
            if (failed(declare(std::get<0>(name_value)->getName(),
                               std::get<1>(name_value))))
                return nullptr;
        }

        builder.setInsertionPointToStart(&entryBlock);

        if (mlir::failed(mlirGen(*funcAST.getBody())))
        {
            function.erase();
            return nullptr;
        }

        ReturnOp returnOp;
        if (!entryBlock.empty())
            returnOp = dyn_cast<ReturnOp>(entryBlock.back());

        if (!returnOp)
        {
            builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
        }
        else if (returnOp.hasOperand())
        {
            function.setType(builder.getFunctionType(function.getType().getInputs(),
                                                     getType(VarType{})));
        }

        return function;
    }

    mlir::Value mlirGen(BinaryExprAST &binop) {
        mlir::Value lhs = mlirGen(*binop.getLHS());
        if (!lhs) return nullptr;
        mlir::Value rhs = mlirGen(*binop.getRHS());
        if (!rhs) return nullptr;
        auto location = loc(binop.loc());

        switch (binop.getOp())
        {
        case '+':
            return builder.create<AddOp>(location, lhs, rhs);
        case '*':
            return builder.create<MulOp>(location, lhs, rhs);
        }

        emitError(location, "invalid binary operator '") << binop.getOp() << "'";
        return nullptr;
    }

    mlir::Value mlirGen(VariableExprAST &expr) {
        if (auto variable = symbolTable.lookup(expr.getName()))
            return variable;

        emitError(loc(expr.loc()), "error: unknown variable '")
            << expr.getName() << "'";
        return nullptr;
    }

   mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
        auto location = loc(ret.loc());

        mlir::Value expr = nullptr;
        if (ret.getExpr().hasValue())
        {
            if (!(expr = mlirGen(*ret.getExpr().getValue())))
                return mlir::failure();
        }

        builder.create<ReturnOp>(location, expr ? makeArrayRef(expr)
                                                : ArrayRef<mlir::Value>());
        return mlir::success();
    }

    mlir::Value mlirGen(LiteralExprAST &lit) {
        auto type = getType(lit.getDims());

        // The attribute is a vector with a floating point value per element
        // (number) in the array, see `collectData()` below for more details.
        std::vector<double> data;
        data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                    std::multiplies<int>()));
        collectData(lit, data);

        // The type of this attribute is tensor of 64-bit floating-point with the
        // shape of the literal.
        mlir::Type elementType = builder.getF64Type();
        auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

        // This is the actual attribute that holds the list of values for this
        // tensor literal.
        auto dataAttribute =
            mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

        // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
        // method.
        return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
    }

    void collectData(ExprAST &expr, std::vector<double> &data) {
        if (auto *lit = dyn_cast<LiteralExprAST>(&expr))
        {
            for (auto &value : lit->getValues())
                collectData(*value, data);
            return;
        }

        assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
        data.push_back(cast<NumberExprAST>(expr).getValue());
    }

    mlir::Value mlirGen(CallExprAST &call) {
        llvm::StringRef callee = call.getCallee();
        auto location = loc(call.loc());

        SmallVector<mlir::Value, 4> operands;
        for (auto &expr : call.getArgs())
        {
            auto arg = mlirGen(*expr);
            if (!arg) return nullptr;
            operands.push_back(arg);
        }

        if (callee == "transpose")
        {
            if (call.getArgs().size() != 1)
            {
                emitError(location, "MLIR codegen encountered an error: toy.transpose "
                                    "does not accept multiple arguments");
                return nullptr;
            }
            return builder.create<TransposeOp>(location, operands[0]);
        }

        return builder.create<GenericCallOp>(location, callee, operands);
    }

    mlir::LogicalResult mlirGen(PrintExprAST &call) {
        auto arg = mlirGen(*call.getArg());
        if (!arg) return mlir::failure();

        builder.create<PrintOp>(loc(call.loc()), arg);
        return mlir::success();
    }

    mlir::Value mlirGen(NumberExprAST &num) {
        return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
    }

    mlir::Value mlirGen(ExprAST &expr) {
        switch (expr.getKind())
        {
        case zyq::ExprAST::Expr_BinOp:
            return mlirGen(cast<BinaryExprAST>(expr));
        case zyq::ExprAST::Expr_Var:
            return mlirGen(cast<VariableExprAST>(expr));
        case zyq::ExprAST::Expr_Literal:
            return mlirGen(cast<LiteralExprAST>(expr));
        case zyq::ExprAST::Expr_Call:
            return mlirGen(cast<CallExprAST>(expr));
        case zyq::ExprAST::Expr_Num:
            return mlirGen(cast<NumberExprAST>(expr));
        default:
            emitError(loc(expr.loc()))
                << "MLIR codegen encountered an unhandled expr kind '"
                << Twine(expr.getKind()) << "'";
            return nullptr;
        }
    }

    mlir::Value mlirGen(VarDeclExprAST &vardecl) {
        auto init = vardecl.getInitVal();
        if (!init)
        {
            emitError(loc(vardecl.loc()),
                          "missing initializer in variable declaration");
            return nullptr;
        }

        mlir::Value value = mlirGen(*init);
        if (!value) return nullptr;

        if (!vardecl.getType().shape.empty())
        {
            value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                              getType(vardecl.getType()),
                                              value);
        }

        if (failed(declare(vardecl.getName(), value)))
            return nullptr;

        return value;
    }

    mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
        ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
        for (auto &expr : blockAST)
        {
            if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get()))
            {
                if (!mlirGen(*vardecl)) return mlir::failure();
                continue;
            }

            if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
                return mlirGen(*ret);

            if (auto *print = dyn_cast<PrintExprAST>(expr.get()))
            {
                if (mlir::failed(mlirGen(*print)))
                    return mlir::success();
                continue;
            }

            if (!mlirGen(*expr)) return mlir::failure();
        }

        return mlir::success();
    }

    mlir::Type getType(ArrayRef<int64_t> shape) {
        if (shape.empty()) 
            return mlir::UnrankedTensorType::get(builder.getF64Type());
        else
            return mlir::RankedTensorType::get(shape, builder.getF64Type());
    }

    mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace


namespace zyq {

mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST)
{
    return MLIRGenImpl(context).mlirGen(moduleAST);
}

}
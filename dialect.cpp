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

#include "dialect.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::zyq;

struct ZYQInlinerInterface : public DialectInlinerInterface
{
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *, Region *,
                         BlockAndValueMapping &) const final {
        return true;
    }

    void handleTerminator(Operation *op,
                          ArrayRef<Value> values_to_repl) const final {
        auto return_op = cast<ReturnOp>(op);

        assert(return_op.getNumOperands() == values_to_repl.size());
        for (const auto &it : llvm::enumerate(return_op.getOperands()))
        {
            values_to_repl[it.index()].replaceAllUsesWith(it.value());
        }
    }

    Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                         Type result_type,
                                         Location conversion_loc) const final {
        return builder.create<CastOp>(conversion_loc, result_type, input);
    }
};

ZYQDialect::ZYQDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<ZYQDialect>())
{
    addOperations<
#define GET_OP_LIST
#include "ops.cc.inc"
    >();

    addInterfaces<ZYQInlinerInterface>();
}

static ParseResult parseBinaryOp(OpAsmParser &parser, OperationState &result)
{
    llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
    llvm::SMLoc operands_loc = parser.getCurrentLocation();

    Type type;
    if (parser.parseOperandList(operands, 2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
    {
        return failure();
    }

    if (FunctionType funcType = type.dyn_cast<FunctionType>())
    {
        if (parser.resolveOperands(operands, funcType.getInputs(),
                                   operands_loc, result.operands))
        {
            return failure();
        }

        result.addTypes(funcType.getResults());
        return success();
    }

    if (parser.resolveOperands(operands, type, result.operands))
    {
        return failure();
    }

    result.addTypes(type);
    return success();
}

static void printBinaryOp(OpAsmPrinter &printer, Operation *op)
{
    printer << op->getName() << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(), [=](Type type) {
            return type == resultType;
        }))
    {
        printer << resultType;
        return;
    }

    printer.printFunctionalType(op->getOperandTypes(),
                                op->getResultTypes());
}

// The OpAsmParser class provides a collection of methods for parsing various punctuation,
// as well as attributes, operands, types, etc. Each of these methods returns a ParseResult.
// This class is a wrapper around LogicalResult that can be converted to a boolean true
// value on failure, or false on success. This allows for easily chaining together a set of
// parser rules. These rules are used to populate an mlir::OperationState similarly to the
// build methods described above.
static ParseResult parseConstantOp(OpAsmParser &parser, OperationState &result)
{
    DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
    {
        return failure();
    }

    result.addTypes(value.getType());
    return success();
}

// The OpAsmPrinter class is a stream that allows for formatting strings, attributes,
// operands, types, etc.
static void printConstantOp(OpAsmPrinter &printer, ConstantOp op)
{
    printer << "zyq.constant ";
    printer.printOptionalAttrDict(op.getAttrs(), {"value"});
    printer << op.value();
}

void ConstantOp::build(OpBuilder &builder, OperationState &state, double value)
{
    auto dataType = RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    ConstantOp::build(builder, state, dataType, dataAttribute);
}

static LogicalResult verify(ConstantOp op)
{
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) return success();

    auto attrType = op.value().getType().cast<TensorType>();
    if (attrType.getRank() != resultType.getRank())
    {
        return op.emitOpError("return type must match the one of the attached"
                              " value attribute: ") << attrType.getRank()
                              << " != " << resultType.getRank();
    }

    for (int dim = 0; dim < attrType.getRank(); ++dim)
    {
        if (attrType.getShape()[dim] != resultType.getShape()[dim])
        {
            return op.emitOpError("return type shape mismatches its attribute"
                                  " at dimension ") << dim << ": "
                                  << resultType.getShape()[dim] << " != "
                                  << attrType.getShape()[dim];
        }
    }

    return success();
}

void AddOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs)
{
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

void AddOp::inferShapes()
{
    getResult().setType(getOperand(0).getType());
}

void CastOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}

void MulOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs)
{
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

void MulOp::inferShapes()
{
    getResult().setType(getOperand(0).getType());
}

void TransposeOp::build(OpBuilder &builder, OperationState &state, Value input)
{
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(input);
}

static LogicalResult verify(TransposeOp op)
{
    auto inputType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getType().dyn_cast<RankedTensorType>();
    if (!inputType || !resultType) return success();

    auto input_shape = inputType.getShape();
    auto output_shape = resultType.getShape();
    if (!std::equal(input_shape.begin(), input_shape.end(),
                    output_shape.rbegin(), output_shape.rend()))
    {
        return op.emitError() << "expected result shape to be a transpose of the input";
    }

    return success();
}

void TransposeOp::inferShapes()
{
    auto array_type = getOperand().getType().cast<RankedTensorType>();
    SmallVector<int64_t, 2> dims(llvm::reverse(array_type.getShape()));
    getResult().setType(RankedTensorType::get(dims, array_type.getElementType()));
}

void GenericCallOp::build(OpBuilder &builder, OperationState &state, StringRef callee,
                          ArrayRef<Value> arguments)
{
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

CallInterfaceCallable GenericCallOp::getCallableForCallee()
{
    return getAttrOfType<SymbolRefAttr>("callee");
}

Operation::operand_range GenericCallOp::getArgOperands()
{
    return inputs();
}

static LogicalResult verify(ReturnOp op)
{
    auto function = cast<FuncOp>(op.getParentOp());

    if (op.getNumOperands() > 1)
        return op.emitOpError() << "expects at most 1 return operand";

    const auto &results = function.getType().getResults();
    if (op.getNumOperands() != results.size())
    {
        return op.emitOpError() << "does not return the same number of values ("
                                << op.getNumOperands() << ") as the enclosing "
                                << "function (" << results.size() << ")";
    }

    if (!op.hasOperand()) return success();

    auto inputType = *op.operand_type_begin();
    auto resultType = results.front();

    if (inputType == resultType || inputType.isa<UnrankedTensorType>() ||
        resultType.isa<UnrankedTensorType>())
    {
        return success();
    }

    return op.emitOpError() << "type of return operand (" << inputType
                            << ") doesn't match function result type ("
                            << resultType << ")"; 
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ops.cc.inc"

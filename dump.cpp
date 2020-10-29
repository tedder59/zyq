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

#include "ast.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace zyq;

template <typename T>
static std::string loc(T* node)
{
    const auto &location = node->loc();
    return (llvm::Twine("@") + *location.file + ":" +
            llvm::Twine(location.line) + ":" +
            llvm::Twine(location.col)).str();
}

#define INDENT()                                        \
    Indent level_(cur_indent_);                         \
    indent()

namespace {

// RAII
struct Indent
{
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
};

class ASTDumper
{
public:
    void dump(ModuleAST *node) {
        INDENT();
        llvm::errs() << "Module:\n";
        for (auto &f : *node) dump(&f);
    }

private:
    void dump(ExprAST *expr) {
        llvm::TypeSwitch<ExprAST *>(expr)
            .Case<BinaryExprAST, CallExprAST, LiteralExprAST,
                  NumberExprAST, PrintExprAST, ReturnExprAST,
                  VarDeclExprAST, VariableExprAST>([&](auto *node) {
                      this->dump(node);
            })
            .Default([&](ExprAST *) {
                INDENT();
                llvm::errs() << "<unknown Expr, kind "
                             << expr->getKind() << ">\n";
            });
    }

    void dump(BinaryExprAST *node) {
        INDENT();
        llvm::errs() << "BinOp: " << node->getOp() << " ["
                     << loc(node) << " \n";
        
        dump(node->getLHS());
        dump(node->getRHS());

        llvm::errs() << "] // Binop\n";
    }

    void dump(CallExprAST *node) {
        INDENT();
        llvm::errs() << "Call '" << node->getCallee()
                     << "' [" << loc(node) << "\n";
        
        for (auto &arg : node->getArgs())
        {
            dump(arg.get());
        }

        indent(); 
        llvm::errs() << "] // Call\n";
    }

    void dump(LiteralExprAST *node) {
        INDENT();
        llvm::errs() << "Literal: ";
        printLiteralHelper(node);
        llvm::errs() << " " << loc(node) << "\n";
    }

    void dump(NumberExprAST *num) {
        INDENT();
        llvm::errs() << num->getValue() << " " << loc(num) << "\n";
    }

    void dump(PrintExprAST *node) {
        INDENT();
        llvm::errs() << "Print [ " << loc(node) << "\n";
        dump(node->getArg());
        indent();
        llvm::errs() << "] // Print\n";
    }

    void dump(ReturnExprAST *node) {
        INDENT();
        llvm::errs() << "Return [" << loc(node) << "\n";

        if (node->getExpr().hasValue())
        {
            dump(*node->getExpr());
        }
        else
        {
            INDENT();
            llvm::errs() << "(void)\n";
        }
        
        llvm::errs() << "] // Return\n";
    }

    void dump(VarDeclExprAST *varDecl) {
        INDENT();
        llvm::errs() << "VarDecl " << varDecl->getName();
        dump(varDecl->getType());
        llvm::errs() << " [" << loc(varDecl) << "\n";
        dump(varDecl->getInitVal());
        indent();
        llvm::errs() << "] // VarDecl\n";
    }

    void dump(VariableExprAST *node) {
        INDENT();
        llvm::errs() << "var: " << node->getName() << " " << loc(node) << "\n";
    }

    void dump(const VarType &type) {
        llvm::errs() << "<";
        llvm::interleaveComma(type.shape, llvm::errs());
        llvm::errs() << ">";
    }    

    void dump(ExprASTList *exprList) {
        INDENT();
        llvm::errs() << "Block {\n";

        for (auto &expr : *exprList)
        {
            dump(expr.get());
        }

        indent();
        llvm::errs() << "} // Block\n";
    }

    void dump(PrototypeAST *node) {
        INDENT();
        llvm::errs() << "Proto '" << node->getName()
                     << "' " << loc(node) << "\n";
        
        indent();
        llvm::errs() << "Params: [";
        llvm::interleaveComma(node->getArgs(), llvm::errs(),
                              [](auto &arg) {
                                  llvm::errs() << arg->getName();
                              });
        llvm::errs() << "]\n";
    }

    void dump(FunctionAST *node) {
        INDENT();
        llvm::errs() << "Function \n";

        dump(node->getProto());
        dump(node->getBody());
    }

    void indent() {
        for (int i = 0; i < cur_indent_; i++)
            llvm::errs() << "  ";
    }

    void printLiteralHelper(ExprAST *node) {
        if (auto num = llvm::dyn_cast<NumberExprAST>(node))
        {
            llvm::errs() << num->getValue();
            return;
        }
        
        auto *literal = llvm::dyn_cast<LiteralExprAST>(node);

        llvm::errs() << "<";
        llvm::interleaveComma(literal->getDims(), llvm::errs());
        llvm::errs() << ">";

        llvm::errs() << "[";
        llvm::interleaveComma(literal->getValues(), llvm::errs(),
                              [&](auto &element) {
                                  this->printLiteralHelper(element.get());
                              });
        llvm::errs() << "]";
    }
    
    int cur_indent_ = 0;
};

}

namespace zyq {

void dump(ModuleAST &module)
{
    ASTDumper().dump(&module);
}

}
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

#ifndef AST_H_
#define AST_H_

#include "lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace zyq {

struct VarType
{
    std::vector<int64_t> shape;
};

class ExprAST
{
public:
    enum ExprASTKind {
        Expr_VarDecl,
        Expr_Return,
        Expr_Num,
        Expr_Literal,
        Expr_Var,
        Expr_BinOp,
        Expr_Call,
        Expr_Print,
    };

    ExprAST(ExprASTKind kind, Location loc)
        : kind_(kind), loc_(loc) {}
    virtual ~ExprAST() = default;

    ExprASTKind getKind() const { return kind_; }
    const Location& loc() { return loc_; }

private:
    const ExprASTKind kind_;
    Location loc_;
};

using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class NumberExprAST : public ExprAST
{
public:
    NumberExprAST(Location loc, double val)
        : ExprAST(Expr_Num, loc), val_(val) {}
    
    double getValue() { return val_; }

    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Num;
    }

private:
    double val_;
};

class LiteralExprAST : public ExprAST
{
public:
    LiteralExprAST(Location loc,
                   std::vector<std::unique_ptr<ExprAST>> values,
                   std::vector<int64_t> dims)
        : ExprAST(Expr_Literal, loc), values_(std::move(values))
        , dims_(std::move(dims)) {}

    llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() {
        return values_;
    }

    llvm::ArrayRef<int64_t> getDims() { return dims_; }

    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Literal;
    }

private:
    std::vector<std::unique_ptr<ExprAST>> values_;
    std::vector<int64_t> dims_;
};

class VariableExprAST : public ExprAST
{
public:
    VariableExprAST(Location loc, llvm::StringRef name)
        : ExprAST(Expr_Var, loc), name_(name) {}
    
    llvm::StringRef getName() { return name_; }

    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Var;
    }

private:
    std::string name_;
};

class VarDeclExprAST : public ExprAST
{
public:
    VarDeclExprAST(Location loc, llvm::StringRef name,
                   VarType type, std::unique_ptr<ExprAST> init_val)
        : ExprAST(Expr_VarDecl, loc), name_(name)
        , type_(std::move(type)), init_val_(std::move(init_val)) {}

    llvm::StringRef getName() { return name_; }
    ExprAST* getInitVal() { return init_val_.get(); }
    const VarType& getType() { return type_; }

    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_VarDecl;
    }

private:
    std::string name_;
    VarType type_;
    std::unique_ptr<ExprAST> init_val_;
};

class ReturnExprAST : public ExprAST
{
public:
    ReturnExprAST(Location loc,
                  llvm::Optional<std::unique_ptr<ExprAST>> expr)
        : ExprAST(Expr_Return, loc), expr_(std::move(expr)) {}
        
    llvm::Optional<ExprAST *> getExpr() {
        if (expr_.hasValue())
            return expr_->get();
            
        return llvm::None;
    }
    
    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Return;
    }

private:
    llvm::Optional<std::unique_ptr<ExprAST>> expr_;
};

class BinaryExprAST : public ExprAST
{
public:
    BinaryExprAST(Location loc, char Op, std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs)
        : ExprAST(Expr_BinOp, loc), op_(Op)
        , lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
        
    char getOp() { return op_; }
    ExprAST *getLHS() { return lhs_.get(); }
    ExprAST *getRHS() { return rhs_.get(); }
    
    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_BinOp;
    }

private:
    char op_;
    std::unique_ptr<ExprAST> lhs_, rhs_;
};

class CallExprAST : public ExprAST
{
public:
    CallExprAST(Location loc, const std::string &callee,
                std::vector<std::unique_ptr<ExprAST>> args)
        : ExprAST(Expr_Call, loc), callee_(callee)
        , args_(std::move(args)) {}
        
    llvm::StringRef getCallee() { return callee_; }
    llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() {
        return args_; 
    }
    
    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Call;
    }

private:
    std::string callee_;
    std::vector<std::unique_ptr<ExprAST>> args_;
};

class PrintExprAST : public ExprAST
{
public:
    PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
        : ExprAST(Expr_Print, loc), arg_(std::move(arg)) {}

    ExprAST *getArg() { return arg_.get(); }
    
    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Print; 
    }

private:
    std::unique_ptr<ExprAST> arg_;
};

class PrototypeAST
{
public:
    PrototypeAST(Location loc, const std::string &name,
                 std::vector<std::unique_ptr<VariableExprAST>> args)
        : loc_(loc), name_(name), args_(std::move(args)) {}

    const Location& loc() { return loc_; }
    llvm::StringRef getName() { return name_; }
    llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() {
        return args_;
    }

private:
    Location loc_;
    std::string name_;
    std::vector<std::unique_ptr<VariableExprAST>> args_;
};

class FunctionAST
{
public:
    FunctionAST(std::unique_ptr<PrototypeAST> proto,
                std::unique_ptr<ExprASTList> body)
        : proto_(std::move(proto)), body_(std::move(body)) {}

    PrototypeAST* getProto() { return proto_.get(); }
    ExprASTList* getBody() { return body_.get(); }

private:
    std::unique_ptr<PrototypeAST> proto_;
    std::unique_ptr<ExprASTList> body_;
};

class ModuleAST
{
    std::vector<FunctionAST> functions_;

public:
    ModuleAST(std::vector<FunctionAST> functions)
        : functions_(std::move(functions)) {}

    auto begin() -> decltype(functions_.begin()) {
        return functions_.begin();
    }

    auto end() -> decltype(functions_.end()) {
        return functions_.end();
    }
};

// void dump(ModuleAST &);

}

#endif // AST_H_
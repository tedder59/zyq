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

#ifndef PARSER_H_
#define PARSER_H_

#include "ast.h"
#include "lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <vector>
#include <utility>

namespace zyq {

class Parser
{
public:
    Parser(Lexer &lexer) : lexer_(lexer) {}

    std::unique_ptr<ModuleAST> parseModule() {
        lexer_.getNextToken();

        std::vector<FunctionAST> functions;
        while (auto f = parseDefinition())
        {
            functions.push_back(std::move(*f));
            if (lexer_.getCurToken() == tok_eof)
                break;
        }

        if (lexer_.getCurToken() != tok_eof)
            return parseError<ModuleAST>("nothing", "at end of module");

        return std::make_unique<ModuleAST>(std::move(functions));
    }

private:
    std::unique_ptr<FunctionAST> parseDefinition() {
        auto proto = parsePrototype();
        if (!proto) return nullptr;

        auto block = parseBlock();
        if (!block) return nullptr;

        return std::make_unique<FunctionAST>(std::move(proto),
                                             std::move(block));
    }

    std::unique_ptr<PrototypeAST> parsePrototype() {
        auto loc = lexer_.getLastLocation();

        if (lexer_.getCurToken() != tok_def)
            return parseError<PrototypeAST>("def", "in prototype");
        lexer_.consume(tok_def);

        if (lexer_.getCurToken() != tok_identifier)
            return parseError<PrototypeAST>("function name", "in prototype");

        std::string func_name(lexer_.getId());
        lexer_.consume(tok_identifier);

        if (lexer_.getCurToken() != '(')
            return parseError<PrototypeAST>("(", "in prototype");
        lexer_.consume(Token('('));

        std::vector<std::unique_ptr<VariableExprAST>> args;
        while (lexer_.getCurToken() != ')')
        {
            std::string name(lexer_.getId());
            auto loc = lexer_.getLastLocation();
            lexer_.consume(tok_identifier);

            auto decl = std::make_unique<VariableExprAST>(std::move(loc), name);
            args.push_back(std::move(decl));

            if (lexer_.getCurToken() != ',')
                break;
            lexer_.consume(Token(','));

            if (lexer_.getCurToken() != tok_identifier)
                return parseError<PrototypeAST>(
                    "identifier", "after ',' in function parameter list"
                );
        }
        
        if (lexer_.getCurToken() != ')')
            return parseError<PrototypeAST>(")", "to end function prototype");
        lexer_.consume(Token(')'));

        return std::make_unique<PrototypeAST>(std::move(loc), func_name, std::move(args));
    }

    std::unique_ptr<ExprASTList> parseBlock() {
        if (lexer_.getCurToken() != '{')
            return parseError<ExprASTList>("{", "to begin block");
        lexer_.consume(Token('{'));

        auto expr_list = std::make_unique<ExprASTList>();
        while (lexer_.getCurToken() == ';') lexer_.consume(Token(';'));

        while (lexer_.getCurToken() != '}' && lexer_.getCurToken() != tok_eof)
        {
            if (lexer_.getCurToken() == tok_var)
            {
                auto var_decl= parseDeclaration();
                if (!var_decl) return nullptr;
                expr_list->push_back(std::move(var_decl));
            }
            else if (lexer_.getCurToken() == tok_return)
            {
                auto ret = parseReturn();
                if (!ret) return nullptr;
                expr_list->push_back(std::move(ret));
            }
            else
            {
                auto expr = parseExpression();
                if (!expr) return nullptr;
                expr_list->push_back(std::move(expr));
            }
            
            if (lexer_.getCurToken() != ';')
                return parseError<ExprASTList>(";", "after expression");

            while (lexer_.getCurToken() == ';') lexer_.consume(Token(';'));
        }
        
        if (lexer_.getCurToken() != '}')
            return parseError<ExprASTList>("}", "to close block");
        lexer_.consume(Token('}'));
        
        return expr_list;
    }

    std::unique_ptr<VarDeclExprAST> parseDeclaration() {
        if (lexer_.getCurToken() != tok_var)
            return parseError<VarDeclExprAST>("var", "to begin declaration");

        auto loc = lexer_.getLastLocation();
        lexer_.getNextToken();

        if (lexer_.getCurToken() != tok_identifier)
            return parseError<VarDeclExprAST>("identified", "after 'var' declaration");
        
        std::string id(lexer_.getId());
        lexer_.getNextToken();

        std::unique_ptr<VarType> type;
        if (lexer_.getCurToken() == '<')
        {
            type = parseType();
            if (!type) return nullptr;
        }

        if (!type)
            type = std::make_unique<VarType>();

        lexer_.consume(Token('='));
        auto expr = parseExpression();

        return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                                std::move(*type), std::move(expr));
    }

    std::unique_ptr<ReturnExprAST> parseReturn() {
        auto loc = lexer_.getLastLocation();
        lexer_.consume(tok_return);

        llvm::Optional<std::unique_ptr<ExprAST>> expr;
        if (lexer_.getCurToken() != ';')
        {
            expr = parseExpression();
            if (!expr) return nullptr;
        }

        return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
    }

    std::unique_ptr<ExprAST> parseExpression() {
        auto lhs = parsePrimary();
        if (!lhs) return nullptr;

        return parseBinOpRHS(0, std::move(lhs));
    }

    std::unique_ptr<VarType> parseType() {
        if (lexer_.getCurToken() != '<')
            return parseError<VarType>("<", "to begin type");
        lexer_.getNextToken(); // eat <

        auto type = std::make_unique<VarType>();
        while (lexer_.getCurToken() == tok_number)
        {
            type->shape.push_back(lexer_.getValue());
            lexer_.getNextToken();
            if (lexer_.getCurToken() == ',')
                lexer_.getNextToken();
        }

        if (lexer_.getCurToken() != '>')
            return parseError<VarType>(">", "to end type");
        lexer_.getNextToken(); // eat >

        return type;
    }

    std::unique_ptr<ExprAST> parsePrimary() {
        switch (lexer_.getCurToken())
        {
        case tok_identifier:
            return parseIdentifierExpr();
        case tok_number:
            return parseNumberExpr();
        case '(':
            return parseParenExpr();
        case '[':
            return parseTensorLiteralExpr();
        case ';':
            return nullptr;
        case '}':
            return nullptr;
        default:
            llvm::errs() << "unknown token '" << lexer_.getCurToken()
                         << "' when expecting an expression\n";
            return nullptr;
        }
    }

    std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                           std::unique_ptr<ExprAST> lhs) {
        while (true)
        {
            int tokPrec = getTokPrecedence();

            if (tokPrec < exprPrec) return lhs;

            int binOp = lexer_.getCurToken();
            lexer_.consume(Token(binOp));
            auto loc = lexer_.getLastLocation();

            auto rhs = parsePrimary();
            if (!rhs)
                return parseError<ExprAST>("expression", "to complete binary operator");

            int nexPrec = getTokPrecedence();
            if (tokPrec < nexPrec)
            {
                rhs = parseBinOpRHS(tokPrec+1, std::move(rhs));
                if (!rhs) return nullptr;
            }

            lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                                  std::move(lhs), std::move(rhs));
        }
    }

    std::unique_ptr<ExprAST> parseIdentifierExpr() {
        std::string name(lexer_.getId());
        
        auto loc = lexer_.getLastLocation();
        lexer_.getNextToken(); // eat identifier.
        
        if (lexer_.getCurToken() != '(') // Simple variable ref.
            return std::make_unique<VariableExprAST>(std::move(loc), name);
        lexer_.consume(Token('('));

        std::vector<std::unique_ptr<ExprAST>> args;
        if (lexer_.getCurToken() != ')')
        {
            while (true)
            {
                if (auto arg = parseExpression())
                    args.push_back(std::move(arg));
                else
                    return nullptr;

                if (lexer_.getCurToken() == ')')
                    break;

                if (lexer_.getCurToken() != ',')
                    return parseError<ExprAST>(", or )", "in argument list");

                lexer_.getNextToken();
            }
        }
        lexer_.consume(Token(')'));

        if (name == "print")
        {
            if (args.size() != 1)
                return parseError<ExprAST>("<single arg>", "as argument to print()");

            return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
        }

        return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
    }

    std::unique_ptr<NumberExprAST> parseNumberExpr() {
        auto loc = lexer_.getLastLocation();
        auto result = std::make_unique<NumberExprAST>(std::move(loc), lexer_.getValue());
        lexer_.consume(tok_number);
        return std::move(result);
    }

    std::unique_ptr<ExprAST> parseParenExpr() {
        lexer_.getNextToken(); // eat (.

        auto v = parseExpression();
        if (!v) return nullptr;

        if (lexer_.getCurToken() != ')')
            return parseError<ExprAST>(")", "to close expression with parentheses");
        lexer_.consume(Token(')'));

        return v;
    }

    std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
        auto loc = lexer_.getLastLocation();
        lexer_.consume(Token('['));

        std::vector<std::unique_ptr<ExprAST>> values;
        std::vector<int64_t> dims;
        do
        {
            if (lexer_.getCurToken() == '[')
            {
                values.push_back(parseTensorLiteralExpr());
                if (!values.back()) return nullptr; // parse error in the nested array.
            }
            else
            {
                if (lexer_.getCurToken() != tok_number)
                    return parseError<ExprAST>("<num> or [", "in literal expression");
                values.push_back(parseNumberExpr());
            }

            if (lexer_.getCurToken() == ']') break;

            if (lexer_.getCurToken() != ',')
                return parseError<ExprAST>("] or ,", "in literal expression");

            lexer_.getNextToken(); // eat ,
        } while (true);

        if (values.empty())
            return parseError<ExprAST>("<something>", "to fill literal expression");
        lexer_.getNextToken(); // eat ]

        dims.push_back(values.size());

        if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
                return llvm::isa<LiteralExprAST>(expr.get());
            } ))
        {
            auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
            if (!firstLiteral)
                return parseError<ExprAST>("uniform well-nested dimensions",
                                           "inside literal expression");

            auto firstDims = firstLiteral->getDims();
            dims.insert(dims.end(), firstDims.begin(), firstDims.end());

            for (auto &expr : values)
            {
                auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
                if (!exprLiteral)
                    return parseError<ExprAST>("uniform well-nested dimensions",
                                               "inside literal expression");
                if (exprLiteral->getDims() != firstDims)
                    return parseError<ExprAST>("uniform well-nested dimensions",
                                               "inside literal expression");
            }
        }

        return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                                std::move(dims));
    }

    int getTokPrecedence() {
        if (!isascii(lexer_.getCurToken()))
            return -1;

        switch (static_cast<char>(lexer_.getCurToken()))
        {
        case '-':
            return 20;
        case '+':
            return 20;
        case '*':
            return 40;
        default:
            return -1;
        }
    }

    template <typename R, typename T, typename U = const char *>
    std::unique_ptr<R> parseError(T &&expected, U &&context="")
    {
        auto curToken = lexer_.getCurToken();
        llvm::errs() << "Parse error (" << lexer_.getLastLocation().line
                     << ", " << lexer_.getLastLocation().col << "): "
                     << "expected '" << expected << "' " << context
                     << " but has Token " << curToken;
        if (isprint(curToken))
            llvm::errs() << " '" << (char)curToken << "'";
        
        llvm::errs() << "\n";
        return nullptr;
    }

    Lexer &lexer_;

};

}

#endif // PARSER_H_
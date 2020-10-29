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

#ifndef LEXER_H_
#define LEXER_H_

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace zyq {

struct Location
{
    std::shared_ptr<std::string> file;
    int line;
    int col;
};

enum Token : int
{
    tok_semicolon           = ';',
    tok_parenthese_open     = '(',
    tok_parenthese_close    = ')',
    tok_bracket_open        = '{',
    tok_bracket_close       = '}',
    tok_sbracket_open       = '[',
    tok_sbracket_close      = ']',

    tok_eof                 = -1,

    tok_return              = -2,
    tok_var                 = -3,
    tok_def                 = -4,

    tok_identifier          = -5,
    tok_number              = -6,
};

class Lexer
{
public:
    Lexer(std::string filename)
        : last_location_({
            std::make_shared<std::string>(std::move(filename)),
            0, 0}) {}
    virtual ~Lexer() = default;

    Token getCurToken() { return cur_tok_; }
    Token getNextToken() { return cur_tok_ = getTok(); }

    void consume(Token tok) {
        assert(tok == cur_tok_ && "consume Token mismatch expectation");
        getNextToken();
    }

    llvm::StringRef getId() {
        assert(cur_tok_ == tok_identifier);
        return identifier_str_;
    }

    double getValue() {
        assert(cur_tok_ == tok_number);
        return num_val_;
    }

    Location getLastLocation() { return last_location_; }

    int getLine() { return cur_line_; }
    int getCol() { return cur_col_; }

private:
    virtual llvm::StringRef readNextLine() = 0;

    int getNextChar() {
        if (cur_line_buffer_.empty()) return EOF;
        
        ++cur_col_;
        auto next_char = cur_line_buffer_.front();
        cur_line_buffer_ = cur_line_buffer_.drop_front();

        if (cur_line_buffer_.empty())
            cur_line_buffer_ = readNextLine();

        if (next_char == '\n')
        {
            ++cur_line_;
            cur_col_ = 0;
        }

        return next_char;
    }

    Token getTok() {
        while (isspace(last_char_))
            last_char_ = Token(getNextChar());
        
        last_location_.line = cur_line_;
        last_location_.col = cur_col_;

        if (isalpha(last_char_))
        {
            identifier_str_ = static_cast<char>(last_char_);
            while (isalnum((last_char_ = Token(getNextChar()))) ||
                   last_char_ == '_')
            {
                identifier_str_ += static_cast<char>(last_char_);
            }
            
            if (identifier_str_ == "return")
                return tok_return;
            if (identifier_str_ == "def")
                return tok_def;
            if (identifier_str_ == "var")
                return tok_var;

            return tok_identifier;
        }

        if (isdigit(last_char_) || last_char_ == '.')
        {
            bool has_point = (last_char_ == '.');

            std::string num_str;
            do {
                num_str += static_cast<char>(last_char_);
                last_char_ = Token(getNextChar());
            } while (isdigit(last_char_) ||
                     (!has_point && last_char_ == '.'));

            num_val_ = strtod(num_str.c_str(), nullptr);

            return tok_number;
        }

        if (last_char_ == '#')
        {
            do
            {
                last_char_ = Token(getNextChar());
            } while (last_char_ != EOF && last_char_ != '\n'
                     && last_char_ != '\r');

            if (last_char_ != EOF) return getTok();
        }

        if (last_char_ == EOF) return tok_eof;
        
        Token this_char = Token(last_char_);
        last_char_ = Token(getNextChar());

        return this_char;
    }

    Token cur_tok_ = tok_eof;

    Location last_location_;

    std::string identifier_str_;
    double num_val_ = 0;

    Token last_char_ = Token(' ');

    int cur_line_ = 0;
    int cur_col_ = 0;

    llvm::StringRef cur_line_buffer_ = "\n";
};

class LexerBuffer final : public Lexer
{
public:
    LexerBuffer(const char *begin, const char *end, std::string filename)
        : Lexer(std::move(filename))
        , current_(begin)
        , end_(end) {}

private:
    llvm::StringRef readNextLine() override {
        auto *begin = current_;
        while (current_ <= end_ && *current_ && *current_ != '\n')
            ++current_;

        if (current_ <= end_ && *current_) ++current_;

        llvm::StringRef line{begin, static_cast<size_t>(current_ - begin)};
        return line;
    }

    const char *current_, *end_;
};

}

#endif // LEXER_H_
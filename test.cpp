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
#include "mlirgen.h"
#include "parser.h"
#include "passes.h"
#include <memory>
#include <fstream>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace zyq;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { ZYQ, MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(ZYQ), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(ZYQ, "zyq", "load the input file as a zyq source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

std::unique_ptr<zyq::ModuleAST> parseInputFile(llvm::StringRef filename)
{
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
        
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

int loadMLIR(llvm::SourceMgr &source_mgr, mlir::MLIRContext &context,
             mlir::OwningModuleRef &module)
{
    if (inputType != InputType::MLIR &&
        !llvm::StringRef(inputFilename).endswith(".mlir"))
    {
        auto module_ast = parseInputFile(inputFilename);
        if (!module_ast) return 6;

        module = mlirGen(context, *module_ast);
        return !module ? 1 : 0;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: "
                     << ec.message() << "\n";
        return -1;
    }

    source_mgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(source_mgr, &context);
    if (!module)
    {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    return 0;
}

int dumpMLIR()
{
    mlir::MLIRContext context(false);
    context.getOrLoadDialect<mlir::zyq::ZYQDialect>();

    mlir::OwningModuleRef module;
    llvm::SourceMgr source_mgr;
    mlir::SourceMgrDiagnosticHandler source_mgr_handler(source_mgr, &context);

    if (int error = loadMLIR(source_mgr, context, module))
        return error;

    mlir::PassManager pm(&context);
    applyPassManagerCLOptions(pm);

    if (enableOpt)
    {
        pm.addPass(mlir::createInlinerPass());

        mlir::OpPassManager &opt_pm = pm.nest<mlir::FuncOp>();
        opt_pm.addPass(mlir::zyq::createShapeInferencePass());
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
    }

    // partial lowering
    {
        pm.addPass(mlir::zyq::createLowerToAffinePass());

        mlir::OpPassManager &opt_pm = pm.nest<mlir::FuncOp>();
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
        opt_pm.addPass(mlir::createLoopFusionPass());
        opt_pm.addPass(mlir::createMemRefDataFlowOptPass());
    }

    if (mlir::failed(pm.run(*module)))
        return 4;

    module->dump();

    return 0;
}

int main(int argc, char **argv)
{
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "zyq compiler\n");

    return dumpMLIR();
}

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

#include "mlir/Pass/Pass.h"
#include "dialect.h"
#include "passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace zyq;

#include "shape_inference_interface.cc.inc"

namespace {

class ShapeInferencePass : public PassWrapper<ShapeInferencePass, FunctionPass>
{
public:
    void runOnFunction() override {
        auto f = getFunction();
        if (f.getName() != "main") return;

        llvm::SmallPtrSet<mlir::Operation*, 16> op_work_list;
        f.walk([&](mlir::Operation *op) {
            if (returnsDynamicShape(op))
                op_work_list.insert(op);
        });

        while (!op_work_list.empty())
        {
            auto nextop = llvm::find_if(op_work_list, allOperandsInferred);
            if (nextop == op_work_list.end()) break;

            Operation *op = *nextop;
            op_work_list.erase(op);

            LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
            if (auto shape_op = dyn_cast<ShapeInference>(op))
            {
                shape_op.inferShapes();
            }
            else
            {
                op->emitError("unable to infer shape of operation without shape"
                              "inference interface");
                return signalPassFailure();
            }
        }

        if (!op_work_list.empty())
        {
            f.emitError("Shape inference failed, ") << op_work_list.size()
                                                    << " operations couldn't be inferred\n";
            signalPassFailure();
        }
    }

    static bool allOperandsInferred(Operation *op) {
        return llvm::all_of(op->getOperandTypes(), [](Type operand_type) {
                return operand_type.isa<RankedTensorType>();
            });
    }

    static bool returnsDynamicShape(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type result_type) {
                return !result_type.isa<RankedTensorType>();
            });
    }
};

}   // anonymous namespace

std::unique_ptr<mlir::Pass> mlir::zyq::createShapeInferencePass()
{
    return std::make_unique<ShapeInferencePass>();
}
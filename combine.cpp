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
#include <numeric>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace zyq;

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp>
{
    SimplifyRedundantTranspose(mlir::MLIRContext *context)
        : OpRewritePattern<TransposeOp>(context, 1) {}

    mlir::LogicalResult matchAndRewrite(TransposeOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value transpose_input = op.getOperand();
        TransposeOp transpose_input_op = transpose_input.getDefiningOp<TransposeOp>();

        if (!transpose_input_op) return failure();

        rewriter.replaceOp(op, {transpose_input_op.getOperand()});
        return success();
    }
};

void TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context)
{
    results.insert<SimplifyRedundantTranspose>(context);
}

namespace {
#include "combine.inc"
}

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands)
{
    return mlir::impl::foldCastOp(*this);
}

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context)
{
    results.insert<ReshapeReshapeOptPattern,
                   RedundantReshapeOptPattern,
                   FoldConstantReshapeOptPattern>(context);
}
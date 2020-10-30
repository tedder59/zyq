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
#include "passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace {

MemRefType convertTensorToMemRef(TensorType type)
{
    assert(type.hasRank() && "expected only ranked shapes");
    return MemRefType::get(type.getShape(),
                           type.getElementType());
}

Value insertAllocAndDealloc(MemRefType type, Location loc,
                            PatternRewriter &rewriter)
{
    auto alloc = rewriter.create<AllocOp>(loc, type);
    auto *parentBlock = alloc.getOperation()->getBlock();
    alloc.getOperation()->moveBefore(&parentBlock->front());

    auto dealloc =  rewriter.create<DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());

    return alloc;
}

using LoopIterationFn = function_ref<Value(OpBuilder &rewriter,
        ValueRange memRefOperands, ValueRange loopIvs)>;

void lowerOpToLoops(Operation *op, ValueRange operands,
                    PatternRewriter &rewriter,
                    LoopIterationFn processIteration)
{
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, tensorType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            Value value_to_store = processIteration(nestedBuilder,
                                                    operands, ivs);
            nestedBuilder.create<AffineStoreOp>(loc, value_to_store, alloc, ivs);
        }
    );

    rewriter.replaceOp(op, alloc);
}

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern
{
    BinaryOpLowering(MLIRContext *ctx)
            : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
                typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);
                auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);
                return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
            });
        return success();
    }
};

using AddOpLowering = BinaryOpLowering<zyq::AddOp, AddFOp>;
using MulOpLowering = BinaryOpLowering<zyq::MulOp, MulFOp>;

struct ConstantOpLowering : public OpRewritePattern<zyq::ConstantOp>
{
    using OpRewritePattern<zyq::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(zyq::ConstantOp op,
                                  PatternRewriter &rewriter) const final {
        DenseElementsAttr constant_value = op.value();
        Location loc = op.getLoc();

        auto tensorType = op.getType().cast<TensorType>();
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        auto value_shape = memRefType.getShape();
        SmallVector<Value, 8> constant_indices;

        if (!value_shape.empty())
        {
            for (auto i : llvm::seq<int64_t>(0, *std::max_element(value_shape.begin(), value_shape.end())))
                constant_indices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
        }
        else
        {
            constant_indices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        }

        SmallVector<Value, 2> indices;
        auto valueIt = constant_value.getValues<FloatAttr>().begin();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            if (dimension == value_shape.size())
            {
                rewriter.create<AffineStoreOp>(
                    loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
                    llvm::makeArrayRef(indices)
                );

                return;
            }

            for (uint64_t i = 0, e = value_shape[dimension]; i != e; ++i)
            {
                indices.push_back(constant_indices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        storeElements(0);
        rewriter.replaceOp(op, alloc);
        
        return success();
    }
};

struct ReturnOpLowering : public OpRewritePattern<zyq::ReturnOp>
{
    using OpRewritePattern<zyq::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(zyq::ReturnOp op,
                                  PatternRewriter &rewriter) const final {
        if (op.hasOperand()) return failure();
        rewriter.replaceOpWithNewOp<ReturnOp>(op);
        return success();
    }
};

struct TransposeOpLowering : public ConversionPattern
{
    TransposeOpLowering(MLIRContext *ctx)
            : ConversionPattern(zyq::TransposeOp::getOperationName(), 1, ctx) {}
        
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final {
        
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands,
                             ValueRange loopIvs) {
                           zyq::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                           Value input = transposeAdaptor.input();

                           SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                           return builder.create<AffineLoadOp>(loc, input, reverseIvs);
        });
        return success();
    }
};

struct ZYQToAffineLoweringPass
        : public PassWrapper<ZYQToAffineLoweringPass, FunctionPass>
{
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<AffineDialect, StandardOpsDialect>();
    }

    void runOnFunction() final {
        auto function = getFunction();
        if (function.getName() != "main") return;

        if (function.getNumArguments() || function.getNumResults())
        {
            function.emitError("expected 'main' to have 0 inputs and 0 results");
            return signalPassFailure();
        }

        ConversionTarget target(getContext());
        target.addLegalDialect<AffineDialect, StandardOpsDialect>();
        target.addIllegalDialect<zyq::ZYQDialect>();
        target.addLegalOp<zyq::PrintOp>();

        OwningRewritePatternList patterns;
        patterns.insert<AddOpLowering, ConstantOpLowering,
                        MulOpLowering, ReturnOpLowering,
                        TransposeOpLowering>(&getContext());
        
        if (failed(applyPartialConversion(getFunction(), target, patterns)))
            signalPassFailure();
    }
};

}

std::unique_ptr<mlir::Pass> mlir::zyq::createLowerToAffinePass()
{
    return std::make_unique<ZYQToAffineLoweringPass>();
}
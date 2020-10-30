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

#ifndef DIALECT_H_
#define DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "shape_inference_interface.h"

namespace mlir {
namespace zyq {

// This is definition of the zyq dialect. A dialect inherits from
// mlir::Dialect and registers custom attributes, operations, and
// types (in ites constructor). It can also override some general
// behavior exposed via virtual methods.
class ZYQDialect : public mlir::Dialect {
public:
    explicit ZYQDialect(mlir::MLIRContext *ctx);

    static llvm::StringRef getDialectNamespace() {
        return "zyq";
    }
};

} // end namespace zyq
} // end namespace mlir

#define GET_OP_CLASSES
#include "ops.h.inc"

#endif // DIALECT_H_
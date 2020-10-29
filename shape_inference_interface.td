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

#ifndef SHPAE_INFERENCE_INTERFACE_TD_
#define SHAPE_INFERENCE_INTERFACE_TD_

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
    let description = [{
        Interface to access a registered method to infer the
        return types for an Operation that can be used during
        type inference.
    }];

    let methods = [
        InterfaceMethod<"Infer an Set the output shape for the current operatoin.",
                        "void", "inferShapes">
    ];
}

#endif // SHAPE_INFERENCE_INTERFACE_TD_
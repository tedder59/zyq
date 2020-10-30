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

#ifndef ZYQ_OPS
#define ZYQ_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "shape_inference_interface.td"

//=========================================================================================//
// zyq dialect
//=========================================================================================//

// Provide a definition of the 'zyq' dialect in the ODS framework so we can define our operations
def ZYQ_Dialect : Dialect {
    let name = "zyq";
    let cppNamespace = "::mlir::zyq";
}

//=========================================================================================//
// zyq traits
//=========================================================================================//

//=========================================================================================//
// zyq attributes
//=========================================================================================//

//=========================================================================================//
// zyq interfaces
//=========================================================================================//

//=========================================================================================//
// zyq Operations
//=========================================================================================//

// Base class for zyq dialect operations. This operation inherits from the base Op class in
// OpBase.td and provides:
//   * The parent dialect of operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ZYQ_Op<string mnemonic, list<OpTrait> traits = []> : Op<ZYQ_Dialect, mnemonic, traits>;

// Define a operation by inheriting from our base ZYQ_Op class above. Provide mnemonic and a
// list of traits for the operation. The constant operation is marked as NoSideEffect as it is
// a pure operation and may be removed if dead.
def ConstantOp : ZYQ_Op<"constant", [NoSideEffect]> {
    // Summary and description. This can be used to auto-generate documentation of operations
    // with our dialect.
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached as an attribute
        ```mlir
            %0 = zyq.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
        ```
    }];

    // The constant operation takes an attribute as the only input.
    let arguments = (ins F64ElementsAttr:$value);

    // The constant operation returns a single value of TensorType.
    let results = (outs F64Tensor);

    // Specify a parser and printer method.
    let parser = [{ return ::parseConstantOp(parser, result); }];
    let printer = [{ return ::printConstantOp(p, *this); }];

    // Add custom build methods for the constant operation. These method populates the state that
    // MLIR uses to create operations, i.e. these are used when using bulider.create<ConstantOp>(...).
    let builders = [
        // Build a constant with a given constant tensor value.
        OpBuilder<"::mlir::DenseElementsAttr value", [{
            build($_builder, $_state, value.getType(), value);
        }]>,

        // Build a constant with a given constant floating-point value.
        OpBuilder<"double value">
    ];

    // Invoke a static verify method to verify this constant operation.
    let verifier = [{ return ::verify(*this); }];
}

def AddOp : ZYQ_Op<"add",
        [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shape of tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let parser = [{ return ::parseBinaryOp(parser, result); }];
    let printer = [{ return ::printBinaryOp(p, *this); }];

    let builders = [
        OpBuilder<"Value lhs, Value rhs">
    ];
}

def CastOp : ZYQ_Op<"cast",
        [NoSideEffect, SameOperandsAndResultShape,
         DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "shape cast operantion";
    let description = [{
        The "cast" operation converts a tensor from one type to an equivalent type
        without changing any data elements. The source and destination types
        must both be tensor types with the same element type. If both are ranked
        then the rank should be the same and static dimensions should match. The
        operation is invalid if converting to a mismatching constant dimension.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);

    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

    let hasFolder = 1;
}

def GenericCallOp : ZYQ_Op<"generic_call",
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to
        be specialized for the shape of its arguments. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
        %4 = toy.generic_call @my_func(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
    let results = (outs F64Tensor);

    // Specialize assembly printing and parsing using a declarative format.
    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<"StringRef callee, ArrayRef<Value> arguments">
    ];
}

def MulOp : ZYQ_Op<"mul",
        [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise multiplication operation";
    let description = [{
        The "add" operation performs element-wise multiplication between two
        tensors. The shape of tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let parser = [{ return ::parseBinaryOp(parser, result); }];
    let printer = [{ return ::printBinaryOp(p, *this); }];

    let builders = [
        OpBuilder<"Value lhs, Value rhs">
    ];
}

def PrintOp : ZYQ_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The print builtin operation prints a given input tensor, and produces no results.
    }];

    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input) ";
}

def ReshapeOp : ZYQ_Op<"reshape", [NoSideEffect]> {
    let summary = "tensor reshape operation";
    let description = [{
        Reshape operation is transforming its input tensor into a new tensor
        with the same number of elements but different shapes. For example:
        ```mlir
            %0 = zyq.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs StaticShapeTensorOf<[F64]>);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let hasCanonicalizer = 1;
}

def ReturnOp : ZYQ_Op<"return", [NoSideEffect,
                                 HasParent<"FuncOp">,
                                 Terminator]> {
    let summary = "return operation";
    let description = [{
        The return operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:
        ```mlir
            func @foo() -> tensor<2xf64> {
                ...
                zyq.return %0 : tensor<2xf64>
            }
        ```
    }];

    let arguments = (ins Variadic<F64Tensor>:$input);

    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    let builders = [
        OpBuilder<"", [{ build($_builder, $_state, llvm::None); }]>
    ];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let verifier = [{ return ::verify(*this); }];
}

def TransposeOp : ZYQ_Op<"transpose",
        [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "transpose operation";
    
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let builders = [
        OpBuilder<"Value input">
    ];

    let verifier = [{ return ::verify(*this); }];

    let hasCanonicalizer = 1;
}

#endif // ZYQ_OPS
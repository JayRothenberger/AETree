κ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??
?
#separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_2/depthwise_kernel
?
7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/depthwise_kernel*&
_output_shapes
:*
dtype0
?
#separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_2/pointwise_kernel
?
7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/pointwise_kernel*&
_output_shapes
:1*
dtype0
?
separable_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*(
shared_nameseparable_conv2d_2/bias

+separable_conv2d_2/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias*
_output_shapes
:1*
dtype0
?
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_3/depthwise_kernel
?
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*&
_output_shapes
:1*
dtype0
?
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*4
shared_name%#separable_conv2d_3/pointwise_kernel
?
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*&
_output_shapes
:11*
dtype0
?
separable_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*(
shared_nameseparable_conv2d_3/bias

+separable_conv2d_3/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias*
_output_shapes
:1*
dtype0
?
#separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_4/depthwise_kernel
?
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*&
_output_shapes
:1*
dtype0
?
#separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*4
shared_name%#separable_conv2d_4/pointwise_kernel
?
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*&
_output_shapes
:11*
dtype0
?
separable_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*(
shared_nameseparable_conv2d_4/bias

+separable_conv2d_4/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias*
_output_shapes
:1*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:1*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value? B?  B? 
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
?
depthwise_kernel
pointwise_kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
depthwise_kernel
pointwise_kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?
(depthwise_kernel
)pointwise_kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
N
0
1
2
3
4
5
(6
)7
*8
/9
010
N
0
1
2
3
4
5
(6
)7
*8
/9
010
 
?
		variables
5layer_metrics
6layer_regularization_losses

7layers

trainable_variables
regularization_losses
8non_trainable_variables
9metrics
 
 
 
 
?
	variables
:layer_metrics
;layer_regularization_losses

<layers
trainable_variables
regularization_losses
=non_trainable_variables
>metrics
yw
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2

0
1
2
 
?
	variables
?layer_metrics
@layer_regularization_losses

Alayers
trainable_variables
regularization_losses
Bnon_trainable_variables
Cmetrics
 
 
 
?
	variables
Dlayer_metrics
Elayer_regularization_losses

Flayers
trainable_variables
regularization_losses
Gnon_trainable_variables
Hmetrics
yw
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2

0
1
2
 
?
 	variables
Ilayer_metrics
Jlayer_regularization_losses

Klayers
!trainable_variables
"regularization_losses
Lnon_trainable_variables
Mmetrics
 
 
 
?
$	variables
Nlayer_metrics
Olayer_regularization_losses

Players
%trainable_variables
&regularization_losses
Qnon_trainable_variables
Rmetrics
yw
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
*2

(0
)1
*2
 
?
+	variables
Slayer_metrics
Tlayer_regularization_losses

Ulayers
,trainable_variables
-regularization_losses
Vnon_trainable_variables
Wmetrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
1	variables
Xlayer_metrics
Ylayer_regularization_losses

Zlayers
2trainable_variables
3regularization_losses
[non_trainable_variables
\metrics
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
y
serving_default_dec_inPlaceholder*'
_output_shapes
:?????????1*
dtype0*
shape:?????????1
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dec_in#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_106026
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_2/bias/Read/ReadVariableOp7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_106305
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_106348??
?
?
.__inference_mnist_dec_var_layer_call_fn_106053

inputs!
unknown:#
	unknown_0:1
	unknown_1:1#
	unknown_2:1#
	unknown_3:11
	unknown_4:1#
	unknown_5:1#
	unknown_6:11
	unknown_7:1#
	unknown_8:1
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1057762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_105629

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_105700

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu?
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_106026

dec_in!
unknown:#
	unknown_0:1
	unknown_1:1#
	unknown_2:1#
	unknown_3:11
	unknown_4:1#
	unknown_5:1#
	unknown_6:11
	unknown_7:1#
	unknown_8:1
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldec_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1055872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_namedec_in
?
L
0__inference_up_sampling2d_1_layer_call_fn_105683

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1056772
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_106249

inputs8
conv2d_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Selu?
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?	
?
3__inference_separable_conv2d_2_layer_call_fn_105616

inputs!
unknown:#
	unknown_0:1
	unknown_1:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1056042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?'
?
__inference__traced_save_106305
file_prefixB
>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_2_bias_read_readvariableopB
>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_3_bias_read_readvariableopB
>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::1:1:1:11:1:1:11:1:1:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
:1: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 	

_output_shapes
:1:,
(
&
_output_shapes
:1: 

_output_shapes
::

_output_shapes
: 
?	
?
3__inference_separable_conv2d_3_layer_call_fn_105664

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1056522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?
?
.__inference_mnist_dec_var_layer_call_fn_106080

inputs!
unknown:#
	unknown_0:1
	unknown_1:1#
	unknown_2:1#
	unknown_3:11
	unknown_4:1#
	unknown_5:1#
	unknown_6:11
	unknown_7:1#
	unknown_8:1
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1058792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_105604

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu?
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
3__inference_separable_conv2d_4_layer_call_fn_105712

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1057002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?'
?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105997

dec_in3
separable_conv2d_2_105968:3
separable_conv2d_2_105970:1'
separable_conv2d_2_105972:13
separable_conv2d_3_105976:13
separable_conv2d_3_105978:11'
separable_conv2d_3_105980:13
separable_conv2d_4_105984:13
separable_conv2d_4_105986:11'
separable_conv2d_4_105988:1)
conv2d_2_105991:1
conv2d_2_105993:
identity?? conv2d_2/StatefulPartitionedCall?*separable_conv2d_2/StatefulPartitionedCall?*separable_conv2d_3/StatefulPartitionedCall?*separable_conv2d_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCalldec_in*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1057332
reshape/PartitionedCall?
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_105968separable_conv2d_2_105970separable_conv2d_2_105972*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1056042,
*separable_conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1056292
up_sampling2d/PartitionedCall?
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_105976separable_conv2d_3_105978separable_conv2d_3_105980*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1056522,
*separable_conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1056772!
up_sampling2d_1/PartitionedCall?
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_105984separable_conv2d_4_105986separable_conv2d_4_105988*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1057002,
*separable_conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_105991conv2d_2_105993*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1057692"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_namedec_in
?'
?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105879

inputs3
separable_conv2d_2_105850:3
separable_conv2d_2_105852:1'
separable_conv2d_2_105854:13
separable_conv2d_3_105858:13
separable_conv2d_3_105860:11'
separable_conv2d_3_105862:13
separable_conv2d_4_105866:13
separable_conv2d_4_105868:11'
separable_conv2d_4_105870:1)
conv2d_2_105873:1
conv2d_2_105875:
identity?? conv2d_2/StatefulPartitionedCall?*separable_conv2d_2/StatefulPartitionedCall?*separable_conv2d_3/StatefulPartitionedCall?*separable_conv2d_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1057332
reshape/PartitionedCall?
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_105850separable_conv2d_2_105852separable_conv2d_2_105854*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1056042,
*separable_conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1056292
up_sampling2d/PartitionedCall?
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_105858separable_conv2d_3_105860separable_conv2d_3_105862*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1056522,
*separable_conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1056772!
up_sampling2d_1/PartitionedCall?
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_105866separable_conv2d_4_105868separable_conv2d_4_105870*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1057002,
*separable_conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_105873conv2d_2_105875*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1057692"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
D
(__inference_reshape_layer_call_fn_106215

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1057332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????1:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_layer_call_fn_105635

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1056292
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_mnist_dec_var_layer_call_fn_105801

dec_in!
unknown:#
	unknown_0:1
	unknown_1:1#
	unknown_2:1#
	unknown_3:11
	unknown_4:1#
	unknown_5:1#
	unknown_6:11
	unknown_7:1#
	unknown_8:1
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldec_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1057762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_namedec_in
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_105769

inputs8
conv2d_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Selu?
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?6
?
"__inference__traced_restore_106348
file_prefixN
4assignvariableop_separable_conv2d_2_depthwise_kernel:P
6assignvariableop_1_separable_conv2d_2_pointwise_kernel:18
*assignvariableop_2_separable_conv2d_2_bias:1P
6assignvariableop_3_separable_conv2d_3_depthwise_kernel:1P
6assignvariableop_4_separable_conv2d_3_pointwise_kernel:118
*assignvariableop_5_separable_conv2d_3_bias:1P
6assignvariableop_6_separable_conv2d_4_depthwise_kernel:1P
6assignvariableop_7_separable_conv2d_4_pointwise_kernel:118
*assignvariableop_8_separable_conv2d_4_bias:1<
"assignvariableop_9_conv2d_2_kernel:1/
!assignvariableop_10_conv2d_2_bias:
identity_12??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp4assignvariableop_separable_conv2d_2_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp6assignvariableop_1_separable_conv2d_2_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_separable_conv2d_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_separable_conv2d_3_depthwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp6assignvariableop_4_separable_conv2d_3_pointwise_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_separable_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_separable_conv2d_4_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_separable_conv2d_4_pointwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_separable_conv2d_4_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_11f
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_12?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?'
?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105964

dec_in3
separable_conv2d_2_105935:3
separable_conv2d_2_105937:1'
separable_conv2d_2_105939:13
separable_conv2d_3_105943:13
separable_conv2d_3_105945:11'
separable_conv2d_3_105947:13
separable_conv2d_4_105951:13
separable_conv2d_4_105953:11'
separable_conv2d_4_105955:1)
conv2d_2_105958:1
conv2d_2_105960:
identity?? conv2d_2/StatefulPartitionedCall?*separable_conv2d_2/StatefulPartitionedCall?*separable_conv2d_3/StatefulPartitionedCall?*separable_conv2d_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCalldec_in*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1057332
reshape/PartitionedCall?
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_105935separable_conv2d_2_105937separable_conv2d_2_105939*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1056042,
*separable_conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1056292
up_sampling2d/PartitionedCall?
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_105943separable_conv2d_3_105945separable_conv2d_3_105947*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1056522,
*separable_conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1056772!
up_sampling2d_1/PartitionedCall?
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_105951separable_conv2d_4_105953separable_conv2d_4_105955*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1057002,
*separable_conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_105958conv2d_2_105960*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1057692"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_namedec_in
?
?
)__inference_conv2d_2_layer_call_fn_106238

inputs!
unknown:1
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1057692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_105587

dec_inc
Imnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource:e
Kmnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1N
@mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource:1c
Imnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource:1e
Kmnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11N
@mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource:1c
Imnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource:1e
Kmnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11N
@mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource:1O
5mnist_dec_var_conv2d_2_conv2d_readvariableop_resource:1D
6mnist_dec_var_conv2d_2_biasadd_readvariableop_resource:
identity??-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp?,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp?7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp?@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp?Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1?7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp?@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp?Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1?7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp?@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp?Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1p
mnist_dec_var/reshape/ShapeShapedec_in*
T0*
_output_shapes
:2
mnist_dec_var/reshape/Shape?
)mnist_dec_var/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)mnist_dec_var/reshape/strided_slice/stack?
+mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_1?
+mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_2?
#mnist_dec_var/reshape/strided_sliceStridedSlice$mnist_dec_var/reshape/Shape:output:02mnist_dec_var/reshape/strided_slice/stack:output:04mnist_dec_var/reshape/strided_slice/stack_1:output:04mnist_dec_var/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#mnist_dec_var/reshape/strided_slice?
%mnist_dec_var/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/1?
%mnist_dec_var/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/2?
%mnist_dec_var/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/3?
#mnist_dec_var/reshape/Reshape/shapePack,mnist_dec_var/reshape/strided_slice:output:0.mnist_dec_var/reshape/Reshape/shape/1:output:0.mnist_dec_var/reshape/Reshape/shape/2:output:0.mnist_dec_var/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#mnist_dec_var/reshape/Reshape/shape?
mnist_dec_var/reshape/ReshapeReshapedec_in,mnist_dec_var/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
mnist_dec_var/reshape/Reshape?
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02B
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp?
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02D
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1?
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7mnist_dec_var/separable_conv2d_2/separable_conv2d/Shape?
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rate?
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative&mnist_dec_var/reshape/Reshape:output:0Hmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise?
1mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_2/separable_conv2d?
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp?
(mnist_dec_var/separable_conv2d_2/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_2/BiasAdd?
%mnist_dec_var/separable_conv2d_2/SeluSelu1mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_2/Selu?
!mnist_dec_var/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!mnist_dec_var/up_sampling2d/Const?
#mnist_dec_var/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#mnist_dec_var/up_sampling2d/Const_1?
mnist_dec_var/up_sampling2d/mulMul*mnist_dec_var/up_sampling2d/Const:output:0,mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2!
mnist_dec_var/up_sampling2d/mul?
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_2/Selu:activations:0#mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2:
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor?
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp?
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1?
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_3/separable_conv2d/Shape?
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rate?
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeImnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise?
1mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_3/separable_conv2d?
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp?
(mnist_dec_var/separable_conv2d_3/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_3/BiasAdd?
%mnist_dec_var/separable_conv2d_3/SeluSelu1mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_3/Selu?
#mnist_dec_var/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#mnist_dec_var/up_sampling2d_1/Const?
%mnist_dec_var/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%mnist_dec_var/up_sampling2d_1/Const_1?
!mnist_dec_var/up_sampling2d_1/mulMul,mnist_dec_var/up_sampling2d_1/Const:output:0.mnist_dec_var/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2#
!mnist_dec_var/up_sampling2d_1/mul?
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_3/Selu:activations:0%mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2<
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor?
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp?
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1?
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_4/separable_conv2d/Shape?
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rate?
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeKmnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise?
1mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_4/separable_conv2d?
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp?
(mnist_dec_var/separable_conv2d_4/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_4/BiasAdd?
%mnist_dec_var/separable_conv2d_4/SeluSelu1mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_4/Selu?
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02.
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp?
mnist_dec_var/conv2d_2/Conv2DConv2D3mnist_dec_var/separable_conv2d_4/Selu:activations:04mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_dec_var/conv2d_2/Conv2D?
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp?
mnist_dec_var/conv2d_2/BiasAddBiasAdd&mnist_dec_var/conv2d_2/Conv2D:output:05mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2 
mnist_dec_var/conv2d_2/BiasAdd?
mnist_dec_var/conv2d_2/SeluSelu'mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
mnist_dec_var/conv2d_2/Selu?
IdentityIdentity)mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp.^mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-^mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp8^mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2^
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp2\
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp2r
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp2?
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp2?
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_12r
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp2?
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp2?
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_12r
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp2?
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp2?
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:O K
'
_output_shapes
:?????????1
 
_user_specified_namedec_in
?'
?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105776

inputs3
separable_conv2d_2_105735:3
separable_conv2d_2_105737:1'
separable_conv2d_2_105739:13
separable_conv2d_3_105743:13
separable_conv2d_3_105745:11'
separable_conv2d_3_105747:13
separable_conv2d_4_105751:13
separable_conv2d_4_105753:11'
separable_conv2d_4_105755:1)
conv2d_2_105770:1
conv2d_2_105772:
identity?? conv2d_2/StatefulPartitionedCall?*separable_conv2d_2/StatefulPartitionedCall?*separable_conv2d_3/StatefulPartitionedCall?*separable_conv2d_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1057332
reshape/PartitionedCall?
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_105735separable_conv2d_2_105737separable_conv2d_2_105739*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1056042,
*separable_conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1056292
up_sampling2d/PartitionedCall?
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_105743separable_conv2d_3_105745separable_conv2d_3_105747*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1056522,
*separable_conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1056772!
up_sampling2d_1/PartitionedCall?
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_105751separable_conv2d_4_105753separable_conv2d_4_105755*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1057002,
*separable_conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_105770conv2d_2_105772*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1057692"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?h
?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_106145

inputsU
;separable_conv2d_2_separable_conv2d_readvariableop_resource:W
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1@
2separable_conv2d_2_biasadd_readvariableop_resource:1U
;separable_conv2d_3_separable_conv2d_readvariableop_resource:1W
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11@
2separable_conv2d_3_biasadd_readvariableop_resource:1U
;separable_conv2d_4_separable_conv2d_readvariableop_resource:1W
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11@
2separable_conv2d_4_biasadd_readvariableop_resource:1A
'conv2d_2_conv2d_readvariableop_resource:16
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?)separable_conv2d_2/BiasAdd/ReadVariableOp?2separable_conv2d_2/separable_conv2d/ReadVariableOp?4separable_conv2d_2/separable_conv2d/ReadVariableOp_1?)separable_conv2d_3/BiasAdd/ReadVariableOp?2separable_conv2d_3/separable_conv2d/ReadVariableOp?4separable_conv2d_3/separable_conv2d/ReadVariableOp_1?)separable_conv2d_4/BiasAdd/ReadVariableOp?2separable_conv2d_4/separable_conv2d/ReadVariableOp?4separable_conv2d_4/separable_conv2d/ReadVariableOp_1T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshape?
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOp?
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_2/separable_conv2d/Shape?
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rate?
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativereshape/Reshape:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwise?
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2d?
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOp?
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_2/BiasAdd?
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_2/Selu{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_2/Selu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOp?
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_3/separable_conv2d/Shape?
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rate?
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise?
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2d?
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOp?
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_3/BiasAdd?
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_3/Selu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_3/Selu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOp?
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_4/separable_conv2d/Shape?
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rate?
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise?
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2d?
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOp?
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/BiasAdd?
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/Selu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D%separable_conv2d_4/Selu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/SeluSeluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Selu~
IdentityIdentityconv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_1:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_106229

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????1:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_105733

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????1:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_105652

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu?
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+???????????????????????????1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????1
 
_user_specified_nameinputs
?
g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_105677

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_mnist_dec_var_layer_call_fn_105931

dec_in!
unknown:#
	unknown_0:1
	unknown_1:1#
	unknown_2:1#
	unknown_3:11
	unknown_4:1#
	unknown_5:1#
	unknown_6:11
	unknown_7:1#
	unknown_8:1
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldec_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1058792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_namedec_in
?h
?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_106210

inputsU
;separable_conv2d_2_separable_conv2d_readvariableop_resource:W
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1@
2separable_conv2d_2_biasadd_readvariableop_resource:1U
;separable_conv2d_3_separable_conv2d_readvariableop_resource:1W
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11@
2separable_conv2d_3_biasadd_readvariableop_resource:1U
;separable_conv2d_4_separable_conv2d_readvariableop_resource:1W
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11@
2separable_conv2d_4_biasadd_readvariableop_resource:1A
'conv2d_2_conv2d_readvariableop_resource:16
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?)separable_conv2d_2/BiasAdd/ReadVariableOp?2separable_conv2d_2/separable_conv2d/ReadVariableOp?4separable_conv2d_2/separable_conv2d/ReadVariableOp_1?)separable_conv2d_3/BiasAdd/ReadVariableOp?2separable_conv2d_3/separable_conv2d/ReadVariableOp?4separable_conv2d_3/separable_conv2d/ReadVariableOp_1?)separable_conv2d_4/BiasAdd/ReadVariableOp?2separable_conv2d_4/separable_conv2d/ReadVariableOp?4separable_conv2d_4/separable_conv2d/ReadVariableOp_1T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshape?
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOp?
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_2/separable_conv2d/Shape?
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rate?
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativereshape/Reshape:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwise?
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2d?
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOp?
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_2/BiasAdd?
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_2/Selu{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_2/Selu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOp?
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_3/separable_conv2d/Shape?
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rate?
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise?
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2d?
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOp?
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_3/BiasAdd?
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_3/Selu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_3/Selu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOp?
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_4/separable_conv2d/Shape?
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rate?
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise?
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2d?
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOp?
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/BiasAdd?
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/Selu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D%separable_conv2d_4/Selu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/SeluSeluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Selu~
IdentityIdentityconv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????1: : : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_1:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
dec_in/
serving_default_dec_in:0?????????1D
conv2d_28
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
]__call__
^_default_save_signature
*_&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
?
depthwise_kernel
pointwise_kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
?
depthwise_kernel
pointwise_kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(depthwise_kernel
)pointwise_kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
n
0
1
2
3
4
5
(6
)7
*8
/9
010"
trackable_list_wrapper
n
0
1
2
3
4
5
(6
)7
*8
/9
010"
trackable_list_wrapper
 "
trackable_list_wrapper
?
		variables
5layer_metrics
6layer_regularization_losses

7layers

trainable_variables
regularization_losses
8non_trainable_variables
9metrics
]__call__
^_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
,
nserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
:layer_metrics
;layer_regularization_losses

<layers
trainable_variables
regularization_losses
=non_trainable_variables
>metrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
=:;2#separable_conv2d_2/depthwise_kernel
=:;12#separable_conv2d_2/pointwise_kernel
%:#12separable_conv2d_2/bias
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
?layer_metrics
@layer_regularization_losses

Alayers
trainable_variables
regularization_losses
Bnon_trainable_variables
Cmetrics
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Dlayer_metrics
Elayer_regularization_losses

Flayers
trainable_variables
regularization_losses
Gnon_trainable_variables
Hmetrics
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
=:;12#separable_conv2d_3/depthwise_kernel
=:;112#separable_conv2d_3/pointwise_kernel
%:#12separable_conv2d_3/bias
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 	variables
Ilayer_metrics
Jlayer_regularization_losses

Klayers
!trainable_variables
"regularization_losses
Lnon_trainable_variables
Mmetrics
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$	variables
Nlayer_metrics
Olayer_regularization_losses

Players
%trainable_variables
&regularization_losses
Qnon_trainable_variables
Rmetrics
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
=:;12#separable_conv2d_4/depthwise_kernel
=:;112#separable_conv2d_4/pointwise_kernel
%:#12separable_conv2d_4/bias
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+	variables
Slayer_metrics
Tlayer_regularization_losses

Ulayers
,trainable_variables
-regularization_losses
Vnon_trainable_variables
Wmetrics
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
):'12conv2d_2/kernel
:2conv2d_2/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1	variables
Xlayer_metrics
Ylayer_regularization_losses

Zlayers
2trainable_variables
3regularization_losses
[non_trainable_variables
\metrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
.__inference_mnist_dec_var_layer_call_fn_105801
.__inference_mnist_dec_var_layer_call_fn_106053
.__inference_mnist_dec_var_layer_call_fn_106080
.__inference_mnist_dec_var_layer_call_fn_105931?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_105587?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *%?"
 ?
dec_in?????????1
?2?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_106145
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_106210
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105964
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105997?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_reshape_layer_call_fn_106215?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_reshape_layer_call_and_return_conditional_losses_106229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_separable_conv2d_2_layer_call_fn_105616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_105604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
.__inference_up_sampling2d_layer_call_fn_105635?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_105629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_separable_conv2d_3_layer_call_fn_105664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????1
?2?
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_105652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????1
?2?
0__inference_up_sampling2d_1_layer_call_fn_105683?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_105677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_separable_conv2d_4_layer_call_fn_105712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????1
?2?
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_105700?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????1
?2?
)__inference_conv2d_2_layer_call_fn_106238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_106249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_106026dec_in"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_105587{()*/0/?,
%?"
 ?
dec_in?????????1
? ";?8
6
conv2d_2*?'
conv2d_2??????????
D__inference_conv2d_2_layer_call_and_return_conditional_losses_106249?/0I?F
??<
:?7
inputs+???????????????????????????1
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv2d_2_layer_call_fn_106238?/0I?F
??<
:?7
inputs+???????????????????????????1
? "2?/+????????????????????????????
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105964?()*/07?4
-?*
 ?
dec_in?????????1
p 

 
? "??<
5?2
0+???????????????????????????
? ?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_105997?()*/07?4
-?*
 ?
dec_in?????????1
p

 
? "??<
5?2
0+???????????????????????????
? ?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_106145u()*/07?4
-?*
 ?
inputs?????????1
p 

 
? "-?*
#? 
0?????????
? ?
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_106210u()*/07?4
-?*
 ?
inputs?????????1
p

 
? "-?*
#? 
0?????????
? ?
.__inference_mnist_dec_var_layer_call_fn_105801z()*/07?4
-?*
 ?
dec_in?????????1
p 

 
? "2?/+????????????????????????????
.__inference_mnist_dec_var_layer_call_fn_105931z()*/07?4
-?*
 ?
dec_in?????????1
p

 
? "2?/+????????????????????????????
.__inference_mnist_dec_var_layer_call_fn_106053z()*/07?4
-?*
 ?
inputs?????????1
p 

 
? "2?/+????????????????????????????
.__inference_mnist_dec_var_layer_call_fn_106080z()*/07?4
-?*
 ?
inputs?????????1
p

 
? "2?/+????????????????????????????
C__inference_reshape_layer_call_and_return_conditional_losses_106229`/?,
%?"
 ?
inputs?????????1
? "-?*
#? 
0?????????
? 
(__inference_reshape_layer_call_fn_106215S/?,
%?"
 ?
inputs?????????1
? " ???????????
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_105604?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????1
? ?
3__inference_separable_conv2d_2_layer_call_fn_105616?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????1?
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_105652?I?F
??<
:?7
inputs+???????????????????????????1
? "??<
5?2
0+???????????????????????????1
? ?
3__inference_separable_conv2d_3_layer_call_fn_105664?I?F
??<
:?7
inputs+???????????????????????????1
? "2?/+???????????????????????????1?
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_105700?()*I?F
??<
:?7
inputs+???????????????????????????1
? "??<
5?2
0+???????????????????????????1
? ?
3__inference_separable_conv2d_4_layer_call_fn_105712?()*I?F
??<
:?7
inputs+???????????????????????????1
? "2?/+???????????????????????????1?
$__inference_signature_wrapper_106026?()*/09?6
? 
/?,
*
dec_in ?
dec_in?????????1";?8
6
conv2d_2*?'
conv2d_2??????????
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_105677?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_1_layer_call_fn_105683?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_105629?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_layer_call_fn_105635?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????
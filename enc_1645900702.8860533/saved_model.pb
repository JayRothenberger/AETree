??
??
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
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
 ?"serve*2.6.02unknown8??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!separable_conv2d/depthwise_kernel
?
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
:*
dtype0
?
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*2
shared_name#!separable_conv2d/pointwise_kernel
?
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*&
_output_shapes
:1*
dtype0
?
separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*&
shared_nameseparable_conv2d/bias
{
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes
:1*
dtype0
?
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_1/depthwise_kernel
?
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
:1*
dtype0
?
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*4
shared_name%#separable_conv2d_1/pointwise_kernel
?
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*&
_output_shapes
:11*
dtype0
?
separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*(
shared_nameseparable_conv2d_1/bias

+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes
:1*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:11*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:1*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
trainable_variables
regularization_losses
		variables

	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?
depthwise_kernel
pointwise_kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?
depthwise_kernel
pointwise_kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
R
&trainable_variables
'regularization_losses
(	variables
)	keras_api
F
0
1
2
3
4
5
6
7
 8
!9
 
F
0
1
2
3
4
5
6
7
 8
!9
?
*layer_metrics
+metrics
trainable_variables
,layer_regularization_losses

-layers
regularization_losses
		variables
.non_trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
/layer_metrics
0metrics
trainable_variables
1layer_regularization_losses

2layers
regularization_losses
	variables
3non_trainable_variables
wu
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
?
4layer_metrics
5metrics
trainable_variables
6layer_regularization_losses

7layers
regularization_losses
	variables
8non_trainable_variables
yw
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
?
9layer_metrics
:metrics
trainable_variables
;layer_regularization_losses

<layers
regularization_losses
	variables
=non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
>layer_metrics
?metrics
"trainable_variables
@layer_regularization_losses

Alayers
#regularization_losses
$	variables
Bnon_trainable_variables
 
 
 
?
Clayer_metrics
Dmetrics
&trainable_variables
Elayer_regularization_losses

Flayers
'regularization_losses
(	variables
Gnon_trainable_variables
 
 
 
*
0
1
2
3
4
5
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
?
serving_default_enc_inPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_enc_inconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_98452
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_98683
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias*
Tin
2*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_98723??
?
?
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_98138

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
2
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
?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98220

inputs&
conv2d_98182:
conv2d_98184:0
separable_conv2d_98187:0
separable_conv2d_98189:1$
separable_conv2d_98191:12
separable_conv2d_1_98194:12
separable_conv2d_1_98196:11&
separable_conv2d_1_98198:1(
conv2d_1_98213:11
conv2d_1_98215:1
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_98182conv2d_98184*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_981812 
conv2d/StatefulPartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_98187separable_conv2d_98189separable_conv2d_98191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_981092*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_98194separable_conv2d_1_98196separable_conv2d_1_98198*
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
GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_981382,
*separable_conv2d_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_98213conv2d_1_98215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_982122"
 conv2d_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_981572*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_mnist_enc_layer_call_fn_98565

inputs!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:1
	unknown_3:1#
	unknown_4:1#
	unknown_5:11
	unknown_6:1#
	unknown_7:11
	unknown_8:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_982202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?R
?
 __inference__wrapped_model_98092

enc_inI
/mnist_enc_conv2d_conv2d_readvariableop_resource:>
0mnist_enc_conv2d_biasadd_readvariableop_resource:]
Cmnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource:_
Emnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource:1H
:mnist_enc_separable_conv2d_biasadd_readvariableop_resource:1_
Emnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource:1a
Gmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11J
<mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource:1K
1mnist_enc_conv2d_1_conv2d_readvariableop_resource:11@
2mnist_enc_conv2d_1_biasadd_readvariableop_resource:1
identity??'mnist_enc/conv2d/BiasAdd/ReadVariableOp?&mnist_enc/conv2d/Conv2D/ReadVariableOp?)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp?(mnist_enc/conv2d_1/Conv2D/ReadVariableOp?1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp?:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp?<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1?3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp?<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp?>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
&mnist_enc/conv2d/Conv2D/ReadVariableOpReadVariableOp/mnist_enc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&mnist_enc/conv2d/Conv2D/ReadVariableOp?
mnist_enc/conv2d/Conv2DConv2Denc_in.mnist_enc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_enc/conv2d/Conv2D?
'mnist_enc/conv2d/BiasAdd/ReadVariableOpReadVariableOp0mnist_enc_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'mnist_enc/conv2d/BiasAdd/ReadVariableOp?
mnist_enc/conv2d/BiasAddBiasAdd mnist_enc/conv2d/Conv2D:output:0/mnist_enc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
mnist_enc/conv2d/BiasAdd?
mnist_enc/conv2d/SeluSelu!mnist_enc/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
mnist_enc/conv2d/Selu?
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpCmnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02<
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp?
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpEmnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02>
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1?
1mnist_enc/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1mnist_enc/separable_conv2d/separable_conv2d/Shape?
9mnist_enc/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2;
9mnist_enc/separable_conv2d/separable_conv2d/dilation_rate?
5mnist_enc/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative#mnist_enc/conv2d/Selu:activations:0Bmnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
27
5mnist_enc/separable_conv2d/separable_conv2d/depthwise?
+mnist_enc/separable_conv2d/separable_conv2dConv2D>mnist_enc/separable_conv2d/separable_conv2d/depthwise:output:0Dmnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2-
+mnist_enc/separable_conv2d/separable_conv2d?
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp:mnist_enc_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype023
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp?
"mnist_enc/separable_conv2d/BiasAddBiasAdd4mnist_enc/separable_conv2d/separable_conv2d:output:09mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12$
"mnist_enc/separable_conv2d/BiasAdd?
mnist_enc/separable_conv2d/SeluSelu+mnist_enc/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12!
mnist_enc/separable_conv2d/Selu?
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpEmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp?
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
3mnist_enc/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_enc/separable_conv2d_1/separable_conv2d/Shape?
;mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rate?
7mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative-mnist_enc/separable_conv2d/Selu:activations:0Dmnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
29
7mnist_enc/separable_conv2d_1/separable_conv2d/depthwise?
-mnist_enc/separable_conv2d_1/separable_conv2dConv2D@mnist_enc/separable_conv2d_1/separable_conv2d/depthwise:output:0Fmnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2/
-mnist_enc/separable_conv2d_1/separable_conv2d?
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp?
$mnist_enc/separable_conv2d_1/BiasAddBiasAdd6mnist_enc/separable_conv2d_1/separable_conv2d:output:0;mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12&
$mnist_enc/separable_conv2d_1/BiasAdd?
!mnist_enc/separable_conv2d_1/SeluSelu-mnist_enc/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12#
!mnist_enc/separable_conv2d_1/Selu?
(mnist_enc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1mnist_enc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02*
(mnist_enc/conv2d_1/Conv2D/ReadVariableOp?
mnist_enc/conv2d_1/Conv2DConv2D/mnist_enc/separable_conv2d_1/Selu:activations:00mnist_enc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
mnist_enc/conv2d_1/Conv2D?
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2mnist_enc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp?
mnist_enc/conv2d_1/BiasAddBiasAdd"mnist_enc/conv2d_1/Conv2D:output:01mnist_enc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
mnist_enc/conv2d_1/BiasAdd?
mnist_enc/conv2d_1/SeluSelu#mnist_enc/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
mnist_enc/conv2d_1/Selu?
9mnist_enc/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9mnist_enc/global_average_pooling2d/Mean/reduction_indices?
'mnist_enc/global_average_pooling2d/MeanMean%mnist_enc/conv2d_1/Selu:activations:0Bmnist_enc/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12)
'mnist_enc/global_average_pooling2d/Mean?
IdentityIdentity0mnist_enc/global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp(^mnist_enc/conv2d/BiasAdd/ReadVariableOp'^mnist_enc/conv2d/Conv2D/ReadVariableOp*^mnist_enc/conv2d_1/BiasAdd/ReadVariableOp)^mnist_enc/conv2d_1/Conv2D/ReadVariableOp2^mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp;^mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp=^mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_14^mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp=^mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp?^mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2R
'mnist_enc/conv2d/BiasAdd/ReadVariableOp'mnist_enc/conv2d/BiasAdd/ReadVariableOp2P
&mnist_enc/conv2d/Conv2D/ReadVariableOp&mnist_enc/conv2d/Conv2D/ReadVariableOp2V
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp2T
(mnist_enc/conv2d_1/Conv2D/ReadVariableOp(mnist_enc/conv2d_1/Conv2D/ReadVariableOp2f
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp2x
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp2|
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_12j
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp2|
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp2?
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?$
?
__inference__traced_save_98683
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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
?: ::::1:1:1:11:1:11:1: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:1: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,	(
&
_output_shapes
:11: 


_output_shapes
:1:

_output_shapes
: 
?
?
)__inference_mnist_enc_layer_call_fn_98590

inputs!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:1
	unknown_3:1#
	unknown_4:1#
	unknown_5:11
	unknown_6:1#
	unknown_7:11
	unknown_8:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_983192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98212

inputs8
conv2d_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:?????????12
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98319

inputs&
conv2d_98293:
conv2d_98295:0
separable_conv2d_98298:0
separable_conv2d_98300:1$
separable_conv2d_98302:12
separable_conv2d_1_98305:12
separable_conv2d_1_98307:11&
separable_conv2d_1_98309:1(
conv2d_1_98312:11
conv2d_1_98314:1
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_98293conv2d_98295*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_981812 
conv2d/StatefulPartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_98298separable_conv2d_98300separable_conv2d_98302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_981092*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_98305separable_conv2d_1_98307separable_conv2d_1_98309*
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
GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_981382,
*separable_conv2d_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_98312conv2d_1_98314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_982122"
 conv2d_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_981572*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_98157

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

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
?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98425

enc_in&
conv2d_98399:
conv2d_98401:0
separable_conv2d_98404:0
separable_conv2d_98406:1$
separable_conv2d_98408:12
separable_conv2d_1_98411:12
separable_conv2d_1_98413:11&
separable_conv2d_1_98415:1(
conv2d_1_98418:11
conv2d_1_98420:1
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_98399conv2d_98401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_981812 
conv2d/StatefulPartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_98404separable_conv2d_98406separable_conv2d_98408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_981092*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_98411separable_conv2d_1_98413separable_conv2d_1_98415*
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
GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_981382,
*separable_conv2d_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_98418conv2d_1_98420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_982122"
 conv2d_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_981572*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?	
?
0__inference_separable_conv2d_layer_call_fn_98121

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
GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_981092
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
?G
?	
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98496

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:S
9separable_conv2d_separable_conv2d_readvariableop_resource:U
;separable_conv2d_separable_conv2d_readvariableop_1_resource:1>
0separable_conv2d_biasadd_readvariableop_resource:1U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:1W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11@
2separable_conv2d_1_biasadd_readvariableop_resource:1A
'conv2d_1_conv2d_readvariableop_resource:116
(conv2d_1_biasadd_readvariableop_resource:1
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'separable_conv2d/BiasAdd/ReadVariableOp?0separable_conv2d/separable_conv2d/ReadVariableOp?2separable_conv2d/separable_conv2d/ReadVariableOp_1?)separable_conv2d_1/BiasAdd/ReadVariableOp?2separable_conv2d_1/separable_conv2d/ReadVariableOp?4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Selu?
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOp?
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1?
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'separable_conv2d/separable_conv2d/Shape?
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rate?
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise?
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d?
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp?
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/BiasAdd?
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/Selu?
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOp?
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_1/separable_conv2d/Shape?
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rate?
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise?
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d?
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOp?
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/BiasAdd?
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/Selu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D%separable_conv2d_1/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
conv2d_1/BiasAdd{
conv2d_1/SeluSeluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
conv2d_1/Selu?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeanconv2d_1/Selu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12
global_average_pooling2d/Mean?
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_98630

inputs!
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_982122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1
 
_user_specified_nameinputs
?G
?	
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98540

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:S
9separable_conv2d_separable_conv2d_readvariableop_resource:U
;separable_conv2d_separable_conv2d_readvariableop_1_resource:1>
0separable_conv2d_biasadd_readvariableop_resource:1U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:1W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11@
2separable_conv2d_1_biasadd_readvariableop_resource:1A
'conv2d_1_conv2d_readvariableop_resource:116
(conv2d_1_biasadd_readvariableop_resource:1
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'separable_conv2d/BiasAdd/ReadVariableOp?0separable_conv2d/separable_conv2d/ReadVariableOp?2separable_conv2d/separable_conv2d/ReadVariableOp_1?)separable_conv2d_1/BiasAdd/ReadVariableOp?2separable_conv2d_1/separable_conv2d/ReadVariableOp?4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Selu?
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOp?
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1?
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'separable_conv2d/separable_conv2d/Shape?
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rate?
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise?
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d?
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp?
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/BiasAdd?
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/Selu?
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOp?
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1?
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_1/separable_conv2d/Shape?
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rate?
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise?
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d?
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOp?
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/BiasAdd?
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/Selu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D%separable_conv2d_1/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
conv2d_1/BiasAdd{
conv2d_1/SeluSeluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
conv2d_1/Selu?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeanconv2d_1/Selu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12
global_average_pooling2d/Mean?
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_mnist_enc_layer_call_fn_98243

enc_in!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:1
	unknown_3:1#
	unknown_4:1#
	unknown_5:11
	unknown_6:1#
	unknown_7:11
	unknown_8:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_982202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?	
?
2__inference_separable_conv2d_1_layer_call_fn_98150

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
GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_981382
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
?
?
&__inference_conv2d_layer_call_fn_98610

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_981812
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
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
T
8__inference_global_average_pooling2d_layer_call_fn_98163

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_981572
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

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
?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98396

enc_in&
conv2d_98370:
conv2d_98372:0
separable_conv2d_98375:0
separable_conv2d_98377:1$
separable_conv2d_98379:12
separable_conv2d_1_98382:12
separable_conv2d_1_98384:11&
separable_conv2d_1_98386:1(
conv2d_1_98389:11
conv2d_1_98391:1
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(separable_conv2d/StatefulPartitionedCall?*separable_conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_98370conv2d_98372*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_981812 
conv2d/StatefulPartitionedCall?
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_98375separable_conv2d_98377separable_conv2d_98379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_981092*
(separable_conv2d/StatefulPartitionedCall?
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_98382separable_conv2d_1_98384separable_conv2d_1_98386*
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
GPU2*0J 8? *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_981382,
*separable_conv2d_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_98389conv2d_1_98391*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_982122"
 conv2d_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_981572*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?1
?
!__inference__traced_restore_98723
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:N
4assignvariableop_2_separable_conv2d_depthwise_kernel:N
4assignvariableop_3_separable_conv2d_pointwise_kernel:16
(assignvariableop_4_separable_conv2d_bias:1P
6assignvariableop_5_separable_conv2d_1_depthwise_kernel:1P
6assignvariableop_6_separable_conv2d_1_pointwise_kernel:118
*assignvariableop_7_separable_conv2d_1_bias:1<
"assignvariableop_8_conv2d_1_kernel:11.
 assignvariableop_9_conv2d_1_bias:1
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_separable_conv2d_depthwise_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp4assignvariableop_3_separable_conv2d_pointwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_separable_conv2d_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_separable_conv2d_1_depthwise_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_separable_conv2d_1_pointwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_separable_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10f
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_11?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
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
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98621

inputs8
conv2d_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:?????????12
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????1
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_98452

enc_in!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:1
	unknown_3:1#
	unknown_4:1#
	unknown_5:11
	unknown_6:1#
	unknown_7:11
	unknown_8:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_980922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_98181

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
T0*/
_output_shapes
:?????????2	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_98109

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
2
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
?
)__inference_mnist_enc_layer_call_fn_98367

enc_in!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:1
	unknown_3:1#
	unknown_4:1#
	unknown_5:11
	unknown_6:1#
	unknown_7:11
	unknown_8:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_983192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_98601

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
T0*/
_output_shapes
:?????????2	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
enc_in7
serving_default_enc_in:0?????????L
global_average_pooling2d0
StatefulPartitionedCall:0?????????1tensorflow/serving/predict:?h
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
trainable_variables
regularization_losses
		variables

	keras_api

signatures
*H&call_and_return_all_conditional_losses
I_default_save_signature
J__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"
_tf_keras_layer
?
depthwise_kernel
pointwise_kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"
_tf_keras_layer
?
depthwise_kernel
pointwise_kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"
_tf_keras_layer
?

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"
_tf_keras_layer
?
&trainable_variables
'regularization_losses
(	variables
)	keras_api
*S&call_and_return_all_conditional_losses
T__call__"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
 8
!9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
 8
!9"
trackable_list_wrapper
?
*layer_metrics
+metrics
trainable_variables
,layer_regularization_losses

-layers
regularization_losses
		variables
.non_trainable_variables
J__call__
I_default_save_signature
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
,
Userving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
/layer_metrics
0metrics
trainable_variables
1layer_regularization_losses

2layers
regularization_losses
	variables
3non_trainable_variables
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
;:92!separable_conv2d/depthwise_kernel
;:912!separable_conv2d/pointwise_kernel
#:!12separable_conv2d/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
4layer_metrics
5metrics
trainable_variables
6layer_regularization_losses

7layers
regularization_losses
	variables
8non_trainable_variables
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
=:;12#separable_conv2d_1/depthwise_kernel
=:;112#separable_conv2d_1/pointwise_kernel
%:#12separable_conv2d_1/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
9layer_metrics
:metrics
trainable_variables
;layer_regularization_losses

<layers
regularization_losses
	variables
=non_trainable_variables
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
):'112conv2d_1/kernel
:12conv2d_1/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
>layer_metrics
?metrics
"trainable_variables
@layer_regularization_losses

Alayers
#regularization_losses
$	variables
Bnon_trainable_variables
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Clayer_metrics
Dmetrics
&trainable_variables
Elayer_regularization_losses

Flayers
'regularization_losses
(	variables
Gnon_trainable_variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98496
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98540
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98396
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98425?
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
 __inference__wrapped_model_98092?
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
annotations? *-?*
(?%
enc_in?????????
?2?
)__inference_mnist_enc_layer_call_fn_98243
)__inference_mnist_enc_layer_call_fn_98565
)__inference_mnist_enc_layer_call_fn_98590
)__inference_mnist_enc_layer_call_fn_98367?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_98601?
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
&__inference_conv2d_layer_call_fn_98610?
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
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_98109?
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
0__inference_separable_conv2d_layer_call_fn_98121?
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
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_98138?
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
2__inference_separable_conv2d_1_layer_call_fn_98150?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98621?
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
(__inference_conv2d_1_layer_call_fn_98630?
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
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_98157?
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
8__inference_global_average_pooling2d_layer_call_fn_98163?
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
?B?
#__inference_signature_wrapper_98452enc_in"?
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
 __inference__wrapped_model_98092?
 !7?4
-?*
(?%
enc_in?????????
? "S?P
N
global_average_pooling2d2?/
global_average_pooling2d?????????1?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98621l !7?4
-?*
(?%
inputs?????????1
? "-?*
#? 
0?????????1
? ?
(__inference_conv2d_1_layer_call_fn_98630_ !7?4
-?*
(?%
inputs?????????1
? " ??????????1?
A__inference_conv2d_layer_call_and_return_conditional_losses_98601l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv2d_layer_call_fn_98610_7?4
-?*
(?%
inputs?????????
? " ???????????
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_98157?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling2d_layer_call_fn_98163wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98396t
 !??<
5?2
(?%
enc_in?????????
p 

 
? "%?"
?
0?????????1
? ?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98425t
 !??<
5?2
(?%
enc_in?????????
p

 
? "%?"
?
0?????????1
? ?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98496t
 !??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????1
? ?
D__inference_mnist_enc_layer_call_and_return_conditional_losses_98540t
 !??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????1
? ?
)__inference_mnist_enc_layer_call_fn_98243g
 !??<
5?2
(?%
enc_in?????????
p 

 
? "??????????1?
)__inference_mnist_enc_layer_call_fn_98367g
 !??<
5?2
(?%
enc_in?????????
p

 
? "??????????1?
)__inference_mnist_enc_layer_call_fn_98565g
 !??<
5?2
(?%
inputs?????????
p 

 
? "??????????1?
)__inference_mnist_enc_layer_call_fn_98590g
 !??<
5?2
(?%
inputs?????????
p

 
? "??????????1?
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_98138?I?F
??<
:?7
inputs+???????????????????????????1
? "??<
5?2
0+???????????????????????????1
? ?
2__inference_separable_conv2d_1_layer_call_fn_98150?I?F
??<
:?7
inputs+???????????????????????????1
? "2?/+???????????????????????????1?
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_98109?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????1
? ?
0__inference_separable_conv2d_layer_call_fn_98121?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????1?
#__inference_signature_wrapper_98452?
 !A?>
? 
7?4
2
enc_in(?%
enc_in?????????"S?P
N
global_average_pooling2d2?/
global_average_pooling2d?????????1
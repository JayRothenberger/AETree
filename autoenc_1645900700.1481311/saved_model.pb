
äÈ
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8èÞ
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
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
¦
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!separable_conv2d/depthwise_kernel

5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
:*
dtype0
¦
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*2
shared_name#!separable_conv2d/pointwise_kernel

5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*&
_output_shapes
:1*
dtype0

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
ª
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_1/depthwise_kernel
£
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
:1*
dtype0
ª
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*4
shared_name%#separable_conv2d_1/pointwise_kernel
£
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*&
_output_shapes
:11*
dtype0

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

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
ª
#separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_2/depthwise_kernel
£
7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/depthwise_kernel*&
_output_shapes
:*
dtype0
ª
#separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_2/pointwise_kernel
£
7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/pointwise_kernel*&
_output_shapes
:1*
dtype0

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
ª
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_3/depthwise_kernel
£
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*&
_output_shapes
:1*
dtype0
ª
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*4
shared_name%#separable_conv2d_3/pointwise_kernel
£
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*&
_output_shapes
:11*
dtype0

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
ª
#separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#separable_conv2d_4/depthwise_kernel
£
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*&
_output_shapes
:1*
dtype0
ª
#separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*4
shared_name%#separable_conv2d_4/pointwise_kernel
£
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*&
_output_shapes
:11*
dtype0

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

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
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
´
(Adam/separable_conv2d/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/m
­
<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/m*&
_output_shapes
:*
dtype0
´
(Adam/separable_conv2d/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/m
­
<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/m*&
_output_shapes
:1*
dtype0

Adam/separable_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*-
shared_nameAdam/separable_conv2d/bias/m

0Adam/separable_conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/m*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_1/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/m
±
>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/m*&
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_1/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/m
±
>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/m*&
_output_shapes
:11*
dtype0

Adam/separable_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_1/bias/m

2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/m*
_output_shapes
:1*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:11*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_2/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_2/depthwise_kernel/m
±
>Adam/separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/depthwise_kernel/m*&
_output_shapes
:*
dtype0
¸
*Adam/separable_conv2d_2/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_2/pointwise_kernel/m
±
>Adam/separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/pointwise_kernel/m*&
_output_shapes
:1*
dtype0

Adam/separable_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_2/bias/m

2Adam/separable_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_2/bias/m*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_3/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_3/depthwise_kernel/m
±
>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/depthwise_kernel/m*&
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_3/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*;
shared_name,*Adam/separable_conv2d_3/pointwise_kernel/m
±
>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/pointwise_kernel/m*&
_output_shapes
:11*
dtype0

Adam/separable_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_3/bias/m

2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_3/bias/m*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_4/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_4/depthwise_kernel/m
±
>Adam/separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/depthwise_kernel/m*&
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_4/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*;
shared_name,*Adam/separable_conv2d_4/pointwise_kernel/m
±
>Adam/separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/pointwise_kernel/m*&
_output_shapes
:11*
dtype0

Adam/separable_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_4/bias/m

2Adam/separable_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_4/bias/m*
_output_shapes
:1*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:1*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
´
(Adam/separable_conv2d/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/v
­
<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/v*&
_output_shapes
:*
dtype0
´
(Adam/separable_conv2d/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/v
­
<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/v*&
_output_shapes
:1*
dtype0

Adam/separable_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*-
shared_nameAdam/separable_conv2d/bias/v

0Adam/separable_conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/v*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_1/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/v
±
>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/v*&
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_1/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/v
±
>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/v*&
_output_shapes
:11*
dtype0

Adam/separable_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_1/bias/v

2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/v*
_output_shapes
:1*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:11*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_2/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_2/depthwise_kernel/v
±
>Adam/separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/depthwise_kernel/v*&
_output_shapes
:*
dtype0
¸
*Adam/separable_conv2d_2/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_2/pointwise_kernel/v
±
>Adam/separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/pointwise_kernel/v*&
_output_shapes
:1*
dtype0

Adam/separable_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_2/bias/v

2Adam/separable_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_2/bias/v*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_3/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_3/depthwise_kernel/v
±
>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/depthwise_kernel/v*&
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_3/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*;
shared_name,*Adam/separable_conv2d_3/pointwise_kernel/v
±
>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/pointwise_kernel/v*&
_output_shapes
:11*
dtype0

Adam/separable_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_3/bias/v

2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_3/bias/v*
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_4/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/separable_conv2d_4/depthwise_kernel/v
±
>Adam/separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/depthwise_kernel/v*&
_output_shapes
:1*
dtype0
¸
*Adam/separable_conv2d_4/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:11*;
shared_name,*Adam/separable_conv2d_4/pointwise_kernel/v
±
>Adam/separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/pointwise_kernel/v*&
_output_shapes
:11*
dtype0

Adam/separable_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/separable_conv2d_4/bias/v

2Adam/separable_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_4/bias/v*
_output_shapes
:1*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:1*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¯u
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*êt
valueàtBÝt BÖt
Ì
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 

layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
trainable_variables
regularization_losses
	variables
	keras_api
¢
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
trainable_variables
regularization_losses
	variables
	keras_api
ä
iter

 beta_1

!beta_2
	"decay
#learning_rate$m¹%mº&m»'m¼(m½)m¾*m¿+mÀ,mÁ-mÂ.mÃ/mÄ0mÅ1mÆ2mÇ3mÈ4mÉ5mÊ6mË7mÌ8mÍ$vÎ%vÏ&vÐ'vÑ(vÒ)vÓ*vÔ+vÕ,vÖ-v×.vØ/vÙ0vÚ1vÛ2vÜ3vÝ4vÞ5vß6và7vá8vâ

$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
 

$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
­
9layer_metrics
:metrics
trainable_variables
;layer_regularization_losses

<layers
regularization_losses
	variables
=non_trainable_variables
 
h

$kernel
%bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api

&depthwise_kernel
'pointwise_kernel
(bias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api

)depthwise_kernel
*pointwise_kernel
+bias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
h

,kernel
-bias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
R
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
F
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
 
F
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
­
Rlayer_metrics
Smetrics
trainable_variables
Tlayer_regularization_losses

Ulayers
regularization_losses
	variables
Vnon_trainable_variables
 
R
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api

.depthwise_kernel
/pointwise_kernel
0bias
[trainable_variables
\regularization_losses
]	variables
^	keras_api
R
_trainable_variables
`regularization_losses
a	variables
b	keras_api

1depthwise_kernel
2pointwise_kernel
3bias
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
R
gtrainable_variables
hregularization_losses
i	variables
j	keras_api

4depthwise_kernel
5pointwise_kernel
6bias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
h

7kernel
8bias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
N
.0
/1
02
13
24
35
46
57
68
79
810
 
N
.0
/1
02
13
24
35
46
57
68
79
810
­
slayer_metrics
tmetrics
trainable_variables
ulayer_regularization_losses

vlayers
regularization_losses
	variables
wnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!separable_conv2d/depthwise_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!separable_conv2d/pointwise_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEseparable_conv2d/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEseparable_conv2d_1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_1/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_1/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEseparable_conv2d_2/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEseparable_conv2d_3/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEseparable_conv2d_4/bias1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_2/kernel1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_2/bias1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
 

x0
 

0
1
2
 

$0
%1
 

$0
%1
­
ylayer_metrics
zmetrics
>trainable_variables
{layer_regularization_losses

|layers
?regularization_losses
@	variables
}non_trainable_variables

&0
'1
(2
 

&0
'1
(2
°
~layer_metrics
metrics
Btrainable_variables
 layer_regularization_losses
layers
Cregularization_losses
D	variables
non_trainable_variables

)0
*1
+2
 

)0
*1
+2
²
layer_metrics
metrics
Ftrainable_variables
 layer_regularization_losses
layers
Gregularization_losses
H	variables
non_trainable_variables

,0
-1
 

,0
-1
²
layer_metrics
metrics
Jtrainable_variables
 layer_regularization_losses
layers
Kregularization_losses
L	variables
non_trainable_variables
 
 
 
²
layer_metrics
metrics
Ntrainable_variables
 layer_regularization_losses
layers
Oregularization_losses
P	variables
non_trainable_variables
 
 
 
*
0

1
2
3
4
5
 
 
 
 
²
layer_metrics
metrics
Wtrainable_variables
 layer_regularization_losses
layers
Xregularization_losses
Y	variables
non_trainable_variables

.0
/1
02
 

.0
/1
02
²
layer_metrics
metrics
[trainable_variables
 layer_regularization_losses
layers
\regularization_losses
]	variables
non_trainable_variables
 
 
 
²
layer_metrics
metrics
_trainable_variables
 layer_regularization_losses
layers
`regularization_losses
a	variables
 non_trainable_variables

10
21
32
 

10
21
32
²
¡layer_metrics
¢metrics
ctrainable_variables
 £layer_regularization_losses
¤layers
dregularization_losses
e	variables
¥non_trainable_variables
 
 
 
²
¦layer_metrics
§metrics
gtrainable_variables
 ¨layer_regularization_losses
©layers
hregularization_losses
i	variables
ªnon_trainable_variables

40
51
62
 

40
51
62
²
«layer_metrics
¬metrics
ktrainable_variables
 ­layer_regularization_losses
®layers
lregularization_losses
m	variables
¯non_trainable_variables

70
81
 

70
81
²
°layer_metrics
±metrics
otrainable_variables
 ²layer_regularization_losses
³layers
pregularization_losses
q	variables
´non_trainable_variables
 
 
 
8
0
1
2
3
4
5
6
7
 
8

µtotal

¶count
·	variables
¸	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

µ0
¶1

·	variables
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/separable_conv2d/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/separable_conv2d_1/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_2/bias/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_3/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_4/bias/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_2/kernel/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_2/bias/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/separable_conv2d/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/separable_conv2d_1/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_2/bias/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_3/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_4/bias/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_2/kernel/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_2/bias/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_enc_inPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_enc_inconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_96588
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_2/bias/Read/ReadVariableOp7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOp0Adam/separable_conv2d/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_2/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOp0Adam/separable_conv2d/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_2/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOpConst*S
TinL
J2H	*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_97526
¦
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/biastotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/m(Adam/separable_conv2d/depthwise_kernel/m(Adam/separable_conv2d/pointwise_kernel/mAdam/separable_conv2d/bias/m*Adam/separable_conv2d_1/depthwise_kernel/m*Adam/separable_conv2d_1/pointwise_kernel/mAdam/separable_conv2d_1/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m*Adam/separable_conv2d_2/depthwise_kernel/m*Adam/separable_conv2d_2/pointwise_kernel/mAdam/separable_conv2d_2/bias/m*Adam/separable_conv2d_3/depthwise_kernel/m*Adam/separable_conv2d_3/pointwise_kernel/mAdam/separable_conv2d_3/bias/m*Adam/separable_conv2d_4/depthwise_kernel/m*Adam/separable_conv2d_4/pointwise_kernel/mAdam/separable_conv2d_4/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/v(Adam/separable_conv2d/depthwise_kernel/v(Adam/separable_conv2d/pointwise_kernel/vAdam/separable_conv2d/bias/v*Adam/separable_conv2d_1/depthwise_kernel/v*Adam/separable_conv2d_1/pointwise_kernel/vAdam/separable_conv2d_1/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v*Adam/separable_conv2d_2/depthwise_kernel/v*Adam/separable_conv2d_2/pointwise_kernel/vAdam/separable_conv2d_2/bias/v*Adam/separable_conv2d_3/depthwise_kernel/v*Adam/separable_conv2d_3/pointwise_kernel/vAdam/separable_conv2d_3/bias/v*Adam/separable_conv2d_4/depthwise_kernel/v*Adam/separable_conv2d_4/pointwise_kernel/vAdam/separable_conv2d_4/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v*R
TinK
I2G*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_97746¯
§	
Ê
0__inference_separable_conv2d_layer_call_fn_95437

inputs!
unknown:#
	unknown_0:1
	unknown_1:1
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_954252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
G
ß	
D__inference_mnist_enc_layer_call_and_return_conditional_losses_96936

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
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢'separable_conv2d/BiasAdd/ReadVariableOp¢0separable_conv2d/separable_conv2d/ReadVariableOp¢2separable_conv2d/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_1/BiasAdd/ReadVariableOp¢2separable_conv2d_1/separable_conv2d/ReadVariableOp¢4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¸
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/Seluæ
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpì
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1«
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'separable_conv2d/separable_conv2d/Shape³
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rateª
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise¥
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d¿
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOpÖ
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d/BiasAdd
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d/Seluì
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpò
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_1/separable_conv2d/Shape·
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rateº
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise­
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2dÅ
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOpÞ
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_1/BiasAdd
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_1/Selu°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÝ
conv2d_1/Conv2DConv2D%separable_conv2d_1/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
conv2d_1/BiasAdd{
conv2d_1/SeluSeluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
conv2d_1/Selu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesÏ
global_average_pooling2d/MeanMeanconv2d_1/Selu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
global_average_pooling2d/Mean
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityú
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·'

D__inference_mnist_dec_layer_call_and_return_conditional_losses_96033

inputs2
separable_conv2d_2_96004:2
separable_conv2d_2_96006:1&
separable_conv2d_2_96008:12
separable_conv2d_3_96012:12
separable_conv2d_3_96014:11&
separable_conv2d_3_96016:12
separable_conv2d_4_96020:12
separable_conv2d_4_96022:11&
separable_conv2d_4_96024:1(
conv2d_2_96027:1
conv2d_2_96029:
identity¢ conv2d_2/StatefulPartitionedCall¢*separable_conv2d_2/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCallÛ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_958872
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96004separable_conv2d_2_96006separable_conv2d_2_96008*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_957582,
*separable_conv2d_2/StatefulPartitionedCall¬
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_957832
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96012separable_conv2d_3_96014separable_conv2d_3_96016*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_958062,
*separable_conv2d_3/StatefulPartitionedCall²
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_958312!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96020separable_conv2d_4_96022separable_conv2d_4_96024*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_958542,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96027conv2d_2_96029*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_959232"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityø
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
«	
Ì
2__inference_separable_conv2d_4_layer_call_fn_95866

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_958542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ë
C
'__inference_reshape_layer_call_fn_97273

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_958872
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

¢
)__inference_mnist_enc_layer_call_fn_95683

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
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_956352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
á
Ç
)__inference_mnist_dec_layer_call_fn_96085

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldec_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_960332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_namedec_in
á
Ç
)__inference_mnist_dec_layer_call_fn_95955

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldec_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_959302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_namedec_in
Ø
K
/__inference_up_sampling2d_1_layer_call_fn_95837

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_958312
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_95473

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«	
Ì
2__inference_separable_conv2d_3_layer_call_fn_95818

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_958062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é
ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95528

inputs8
conv2d_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
·'

D__inference_mnist_dec_layer_call_and_return_conditional_losses_95930

inputs2
separable_conv2d_2_95889:2
separable_conv2d_2_95891:1&
separable_conv2d_2_95893:12
separable_conv2d_3_95897:12
separable_conv2d_3_95899:11&
separable_conv2d_3_95901:12
separable_conv2d_4_95905:12
separable_conv2d_4_95907:11&
separable_conv2d_4_95909:1(
conv2d_2_95924:1
conv2d_2_95926:
identity¢ conv2d_2/StatefulPartitionedCall¢*separable_conv2d_2/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCallÛ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_958872
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_95889separable_conv2d_2_95891separable_conv2d_2_95893*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_957582,
*separable_conv2d_2/StatefulPartitionedCall¬
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_957832
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_95897separable_conv2d_3_95899separable_conv2d_3_95901*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_958062,
*separable_conv2d_3/StatefulPartitionedCall²
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_958312!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_95905separable_conv2d_4_95907separable_conv2d_4_95909*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_958542,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_95924conv2d_2_95926*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_959232"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityø
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¦

M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_95806

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identity­
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


#__inference_signature_wrapper_96588

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
	unknown_8:1#
	unknown_9:$

unknown_10:1

unknown_11:1$

unknown_12:1$

unknown_13:11

unknown_14:1$

unknown_15:1$

unknown_16:11

unknown_17:1$

unknown_18:1

unknown_19:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_954082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
«	
Ì
2__inference_separable_conv2d_1_layer_call_fn_95466

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_954542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Û

(__inference_mnist_ae_layer_call_fn_96845

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
	unknown_8:1#
	unknown_9:$

unknown_10:1

unknown_11:1$

unknown_12:1$

unknown_13:11

unknown_14:1$

unknown_15:1$

unknown_16:11

unknown_17:1$

unknown_18:1

unknown_19:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_mnist_ae_layer_call_and_return_conditional_losses_962032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
G
ß	
D__inference_mnist_enc_layer_call_and_return_conditional_losses_96980

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
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢'separable_conv2d/BiasAdd/ReadVariableOp¢0separable_conv2d/separable_conv2d/ReadVariableOp¢2separable_conv2d/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_1/BiasAdd/ReadVariableOp¢2separable_conv2d_1/separable_conv2d/ReadVariableOp¢4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¸
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/Seluæ
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpì
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1«
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'separable_conv2d/separable_conv2d/Shape³
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rateª
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise¥
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d¿
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOpÖ
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d/BiasAdd
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d/Seluì
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpò
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_1/separable_conv2d/Shape·
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rateº
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise­
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2dÅ
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOpÞ
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_1/BiasAdd
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_1/Selu°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÝ
conv2d_1/Conv2DConv2D%separable_conv2d_1/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
conv2d_1/BiasAdd{
conv2d_1/SeluSeluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
conv2d_1/Selu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesÏ
global_average_pooling2d/MeanMeanconv2d_1/Selu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
global_average_pooling2d/Mean
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityú
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
¥
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95635

inputs&
conv2d_95609:
conv2d_95611:0
separable_conv2d_95614:0
separable_conv2d_95616:1$
separable_conv2d_95618:12
separable_conv2d_1_95621:12
separable_conv2d_1_95623:11&
separable_conv2d_1_95625:1(
conv2d_1_95628:11
conv2d_1_95630:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_95609conv2d_95611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_954972 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_95614separable_conv2d_95616separable_conv2d_95618*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_954252*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95621separable_conv2d_1_95623separable_conv2d_1_95625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_954542,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_95628conv2d_1_95630*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_955282"
 conv2d_1/StatefulPartitionedCall©
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_954732*
(global_average_pooling2d/PartitionedCall
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityê
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_95923

inputs8
conv2d_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¦

M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_95758

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identity­
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·'

D__inference_mnist_dec_layer_call_and_return_conditional_losses_96118

dec_in2
separable_conv2d_2_96089:2
separable_conv2d_2_96091:1&
separable_conv2d_2_96093:12
separable_conv2d_3_96097:12
separable_conv2d_3_96099:11&
separable_conv2d_3_96101:12
separable_conv2d_4_96105:12
separable_conv2d_4_96107:11&
separable_conv2d_4_96109:1(
conv2d_2_96112:1
conv2d_2_96114:
identity¢ conv2d_2/StatefulPartitionedCall¢*separable_conv2d_2/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCallÛ
reshape/PartitionedCallPartitionedCalldec_in*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_958872
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96089separable_conv2d_2_96091separable_conv2d_2_96093*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_957582,
*separable_conv2d_2/StatefulPartitionedCall¬
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_957832
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96097separable_conv2d_3_96099separable_conv2d_3_96101*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_958062,
*separable_conv2d_3/StatefulPartitionedCall²
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_958312!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96105separable_conv2d_4_96107separable_conv2d_4_96109*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_958542,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96112conv2d_2_96114*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_959232"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityø
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_namedec_in
Û

(__inference_mnist_ae_layer_call_fn_96437

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
	unknown_8:1#
	unknown_9:$

unknown_10:1

unknown_11:1$

unknown_12:1$

unknown_13:11

unknown_14:1$

unknown_15:1$

unknown_16:11

unknown_17:1$

unknown_18:1

unknown_19:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_mnist_ae_layer_call_and_return_conditional_losses_963452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in

^
B__inference_reshape_layer_call_and_return_conditional_losses_97268

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
strided_slice/stack_2â
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
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

¢
)__inference_mnist_enc_layer_call_fn_95559

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
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_955362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
«
­"
__inference__traced_save_97526
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopB
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
(savev2_conv2d_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableopG
Csavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopG
Csavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop;
7savev2_adam_separable_conv2d_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_2_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_2_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_2_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_3_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_4_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_4_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableopG
Csavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopG
Csavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop;
7savev2_adam_separable_conv2d_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_2_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_2_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_2_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_3_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_4_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_4_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameù%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*%
value%Bþ$GB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*£
valueBGB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesª!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop7savev2_adam_separable_conv2d_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_2_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_3_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop7savev2_adam_separable_conv2d_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_2_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_3_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *U
dtypesK
I2G	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*õ
_input_shapesã
à: : : : : : ::::1:1:1:11:1:11:1::1:1:1:11:1:1:11:1:1:: : ::::1:1:1:11:1:11:1::1:1:1:11:1:1:11:1:1:::::1:1:1:11:1:11:1::1:1:1:11:1:1:11:1:1:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,	(
&
_output_shapes
:1: 


_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
::,(
&
_output_shapes
:1: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
:1: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::, (
&
_output_shapes
:1: !

_output_shapes
:1:,"(
&
_output_shapes
:1:,#(
&
_output_shapes
:11: $

_output_shapes
:1:,%(
&
_output_shapes
:11: &

_output_shapes
:1:,'(
&
_output_shapes
::,((
&
_output_shapes
:1: )

_output_shapes
:1:,*(
&
_output_shapes
:1:,+(
&
_output_shapes
:11: ,

_output_shapes
:1:,-(
&
_output_shapes
:1:,.(
&
_output_shapes
:11: /

_output_shapes
:1:,0(
&
_output_shapes
:1: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
::,5(
&
_output_shapes
:1: 6

_output_shapes
:1:,7(
&
_output_shapes
:1:,8(
&
_output_shapes
:11: 9

_output_shapes
:1:,:(
&
_output_shapes
:11: ;

_output_shapes
:1:,<(
&
_output_shapes
::,=(
&
_output_shapes
:1: >

_output_shapes
:1:,?(
&
_output_shapes
:1:,@(
&
_output_shapes
:11: A

_output_shapes
:1:,B(
&
_output_shapes
:1:,C(
&
_output_shapes
:11: D

_output_shapes
:1:,E(
&
_output_shapes
:1: F

_output_shapes
::G

_output_shapes
: 
é
¥
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95712

enc_in&
conv2d_95686:
conv2d_95688:0
separable_conv2d_95691:0
separable_conv2d_95693:1$
separable_conv2d_95695:12
separable_conv2d_1_95698:12
separable_conv2d_1_95700:11&
separable_conv2d_1_95702:1(
conv2d_1_95705:11
conv2d_1_95707:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_95686conv2d_95688*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_954972 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_95691separable_conv2d_95693separable_conv2d_95695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_954252*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95698separable_conv2d_1_95700separable_conv2d_1_95702*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_954542,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_95705conv2d_1_95707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_955282"
 conv2d_1/StatefulPartitionedCall©
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_954732*
(global_average_pooling2d/PartitionedCall
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityê
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in

^
B__inference_reshape_layer_call_and_return_conditional_losses_95887

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
strided_slice/stack_2â
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
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ô
Ñ
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96345

inputs)
mnist_enc_96300:
mnist_enc_96302:)
mnist_enc_96304:)
mnist_enc_96306:1
mnist_enc_96308:1)
mnist_enc_96310:1)
mnist_enc_96312:11
mnist_enc_96314:1)
mnist_enc_96316:11
mnist_enc_96318:1)
mnist_dec_96321:)
mnist_dec_96323:1
mnist_dec_96325:1)
mnist_dec_96327:1)
mnist_dec_96329:11
mnist_dec_96331:1)
mnist_dec_96333:1)
mnist_dec_96335:11
mnist_dec_96337:1)
mnist_dec_96339:1
mnist_dec_96341:
identity¢!mnist_dec/StatefulPartitionedCall¢!mnist_enc/StatefulPartitionedCall±
!mnist_enc/StatefulPartitionedCallStatefulPartitionedCallinputsmnist_enc_96300mnist_enc_96302mnist_enc_96304mnist_enc_96306mnist_enc_96308mnist_enc_96310mnist_enc_96312mnist_enc_96314mnist_enc_96316mnist_enc_96318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_956352#
!mnist_enc/StatefulPartitionedCall
!mnist_dec/StatefulPartitionedCallStatefulPartitionedCall*mnist_enc/StatefulPartitionedCall:output:0mnist_dec_96321mnist_dec_96323mnist_dec_96325mnist_dec_96327mnist_dec_96329mnist_dec_96331mnist_dec_96333mnist_dec_96335mnist_dec_96337mnist_dec_96339mnist_dec_96341*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_960332#
!mnist_dec/StatefulPartitionedCall
IdentityIdentity*mnist_dec/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^mnist_dec/StatefulPartitionedCall"^mnist_enc/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2F
!mnist_dec/StatefulPartitionedCall!mnist_dec/StatefulPartitionedCall2F
!mnist_enc/StatefulPartitionedCall!mnist_enc/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
Ç
)__inference_mnist_dec_layer_call_fn_97214

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_960332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
±
f
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_95831

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ðh
¿
D__inference_mnist_dec_layer_call_and_return_conditional_losses_97095

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
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢)separable_conv2d_2/BiasAdd/ReadVariableOp¢2separable_conv2d_2/separable_conv2d/ReadVariableOp¢4separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_3/BiasAdd/ReadVariableOp¢2separable_conv2d_3/separable_conv2d/ReadVariableOp¢4separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_4/BiasAdd/ReadVariableOp¢2separable_conv2d_4/separable_conv2d/ReadVariableOp¢4separable_conv2d_4/separable_conv2d/ReadVariableOp_1T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shape/3ê
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshapeì
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpò
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_2/separable_conv2d/Shape·
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rate¯
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativereshape/Reshape:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwise­
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2dÅ
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOpÞ
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_2/BiasAdd
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
up_sampling2d/Const_1
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_2/Selu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborì
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpò
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_3/separable_conv2d/Shape·
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rateÒ
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise­
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2dÅ
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpÞ
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_3/BiasAdd
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_3/Selu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_3/Selu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborì
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpò
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_4/separable_conv2d/Shape·
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rateÔ
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise­
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2dÅ
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpÞ
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_4/BiasAdd
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_4/Selu°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÝ
conv2d_2/Conv2DConv2D%separable_conv2d_4/Selu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd{
conv2d_2/SeluSeluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Selu~
IdentityIdentityconv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÙ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 2B
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
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Û

(__inference_mnist_ae_layer_call_fn_96892

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
	unknown_8:1#
	unknown_9:$

unknown_10:1

unknown_11:1$

unknown_12:1$

unknown_13:11

unknown_14:1$

unknown_15:1$

unknown_16:11

unknown_17:1$

unknown_18:1

unknown_19:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_mnist_ae_layer_call_and_return_conditional_losses_963452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¢
)__inference_mnist_enc_layer_call_fn_97030

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
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_956352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97284

inputs8
conv2d_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ô
I
-__inference_up_sampling2d_layer_call_fn_95789

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_957832
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¢
)__inference_mnist_enc_layer_call_fn_97005

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
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_955362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_95854

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identity­
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
È
Þ
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96693

inputsI
/mnist_enc_conv2d_conv2d_readvariableop_resource:>
0mnist_enc_conv2d_biasadd_readvariableop_resource:]
Cmnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource:_
Emnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource:1H
:mnist_enc_separable_conv2d_biasadd_readvariableop_resource:1_
Emnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource:1a
Gmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11J
<mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource:1K
1mnist_enc_conv2d_1_conv2d_readvariableop_resource:11@
2mnist_enc_conv2d_1_biasadd_readvariableop_resource:1_
Emnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_resource:a
Gmnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1J
<mnist_dec_separable_conv2d_2_biasadd_readvariableop_resource:1_
Emnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_resource:1a
Gmnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11J
<mnist_dec_separable_conv2d_3_biasadd_readvariableop_resource:1_
Emnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_resource:1a
Gmnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11J
<mnist_dec_separable_conv2d_4_biasadd_readvariableop_resource:1K
1mnist_dec_conv2d_2_conv2d_readvariableop_resource:1@
2mnist_dec_conv2d_2_biasadd_readvariableop_resource:
identity¢)mnist_dec/conv2d_2/BiasAdd/ReadVariableOp¢(mnist_dec/conv2d_2/Conv2D/ReadVariableOp¢3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp¢<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp¢>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp¢<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp¢>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp¢<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp¢>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1¢'mnist_enc/conv2d/BiasAdd/ReadVariableOp¢&mnist_enc/conv2d/Conv2D/ReadVariableOp¢)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp¢(mnist_enc/conv2d_1/Conv2D/ReadVariableOp¢1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp¢:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp¢<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1¢3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp¢<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp¢>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1È
&mnist_enc/conv2d/Conv2D/ReadVariableOpReadVariableOp/mnist_enc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&mnist_enc/conv2d/Conv2D/ReadVariableOpÖ
mnist_enc/conv2d/Conv2DConv2Dinputs.mnist_enc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_enc/conv2d/Conv2D¿
'mnist_enc/conv2d/BiasAdd/ReadVariableOpReadVariableOp0mnist_enc_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'mnist_enc/conv2d/BiasAdd/ReadVariableOpÌ
mnist_enc/conv2d/BiasAddBiasAdd mnist_enc/conv2d/Conv2D:output:0/mnist_enc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_enc/conv2d/BiasAdd
mnist_enc/conv2d/SeluSelu!mnist_enc/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_enc/conv2d/Selu
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpCmnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02<
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpEmnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02>
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1¿
1mnist_enc/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1mnist_enc/separable_conv2d/separable_conv2d/ShapeÇ
9mnist_enc/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2;
9mnist_enc/separable_conv2d/separable_conv2d/dilation_rateÒ
5mnist_enc/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative#mnist_enc/conv2d/Selu:activations:0Bmnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
27
5mnist_enc/separable_conv2d/separable_conv2d/depthwiseÍ
+mnist_enc/separable_conv2d/separable_conv2dConv2D>mnist_enc/separable_conv2d/separable_conv2d/depthwise:output:0Dmnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2-
+mnist_enc/separable_conv2d/separable_conv2dÝ
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp:mnist_enc_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype023
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpþ
"mnist_enc/separable_conv2d/BiasAddBiasAdd4mnist_enc/separable_conv2d/separable_conv2d:output:09mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12$
"mnist_enc/separable_conv2d/BiasAdd±
mnist_enc/separable_conv2d/SeluSelu+mnist_enc/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12!
mnist_enc/separable_conv2d/Selu
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpEmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ã
3mnist_enc/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_enc/separable_conv2d_1/separable_conv2d/ShapeË
;mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rateâ
7mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative-mnist_enc/separable_conv2d/Selu:activations:0Dmnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
29
7mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseÕ
-mnist_enc/separable_conv2d_1/separable_conv2dConv2D@mnist_enc/separable_conv2d_1/separable_conv2d/depthwise:output:0Fmnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_enc/separable_conv2d_1/separable_conv2dã
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp
$mnist_enc/separable_conv2d_1/BiasAddBiasAdd6mnist_enc/separable_conv2d_1/separable_conv2d:output:0;mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_enc/separable_conv2d_1/BiasAdd·
!mnist_enc/separable_conv2d_1/SeluSelu-mnist_enc/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_enc/separable_conv2d_1/SeluÎ
(mnist_enc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1mnist_enc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02*
(mnist_enc/conv2d_1/Conv2D/ReadVariableOp
mnist_enc/conv2d_1/Conv2DConv2D/mnist_enc/separable_conv2d_1/Selu:activations:00mnist_enc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
mnist_enc/conv2d_1/Conv2DÅ
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2mnist_enc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOpÔ
mnist_enc/conv2d_1/BiasAddBiasAdd"mnist_enc/conv2d_1/Conv2D:output:01mnist_enc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc/conv2d_1/BiasAdd
mnist_enc/conv2d_1/SeluSelu#mnist_enc/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc/conv2d_1/SeluÇ
9mnist_enc/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9mnist_enc/global_average_pooling2d/Mean/reduction_indices÷
'mnist_enc/global_average_pooling2d/MeanMean%mnist_enc/conv2d_1/Selu:activations:0Bmnist_enc/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_enc/global_average_pooling2d/Mean
mnist_dec/reshape/ShapeShape0mnist_enc/global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:2
mnist_dec/reshape/Shape
%mnist_dec/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%mnist_dec/reshape/strided_slice/stack
'mnist_dec/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'mnist_dec/reshape/strided_slice/stack_1
'mnist_dec/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'mnist_dec/reshape/strided_slice/stack_2Î
mnist_dec/reshape/strided_sliceStridedSlice mnist_dec/reshape/Shape:output:0.mnist_dec/reshape/strided_slice/stack:output:00mnist_dec/reshape/strided_slice/stack_1:output:00mnist_dec/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
mnist_dec/reshape/strided_slice
!mnist_dec/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!mnist_dec/reshape/Reshape/shape/1
!mnist_dec/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!mnist_dec/reshape/Reshape/shape/2
!mnist_dec/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!mnist_dec/reshape/Reshape/shape/3¦
mnist_dec/reshape/Reshape/shapePack(mnist_dec/reshape/strided_slice:output:0*mnist_dec/reshape/Reshape/shape/1:output:0*mnist_dec/reshape/Reshape/shape/2:output:0*mnist_dec/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
mnist_dec/reshape/Reshape/shape×
mnist_dec/reshape/ReshapeReshape0mnist_enc/global_average_pooling2d/Mean:output:0(mnist_dec/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec/reshape/Reshape
<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpEmnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp
>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02@
>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ã
3mnist_dec/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            25
3mnist_dec/separable_conv2d_2/separable_conv2d/ShapeË
;mnist_dec/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_dec/separable_conv2d_2/separable_conv2d/dilation_rate×
7mnist_dec/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative"mnist_dec/reshape/Reshape:output:0Dmnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
29
7mnist_dec/separable_conv2d_2/separable_conv2d/depthwiseÕ
-mnist_dec/separable_conv2d_2/separable_conv2dConv2D@mnist_dec/separable_conv2d_2/separable_conv2d/depthwise:output:0Fmnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_dec/separable_conv2d_2/separable_conv2dã
3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<mnist_dec_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp
$mnist_dec/separable_conv2d_2/BiasAddBiasAdd6mnist_dec/separable_conv2d_2/separable_conv2d:output:0;mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_dec/separable_conv2d_2/BiasAdd·
!mnist_dec/separable_conv2d_2/SeluSelu-mnist_dec/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_dec/separable_conv2d_2/Selu
mnist_dec/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
mnist_dec/up_sampling2d/Const
mnist_dec/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2!
mnist_dec/up_sampling2d/Const_1¸
mnist_dec/up_sampling2d/mulMul&mnist_dec/up_sampling2d/Const:output:0(mnist_dec/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
mnist_dec/up_sampling2d/mul«
4mnist_dec/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor/mnist_dec/separable_conv2d_2/Selu:activations:0mnist_dec/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(26
4mnist_dec/up_sampling2d/resize/ResizeNearestNeighbor
<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpEmnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp
>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ã
3mnist_dec/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_dec/separable_conv2d_3/separable_conv2d/ShapeË
;mnist_dec/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_dec/separable_conv2d_3/separable_conv2d/dilation_rateú
7mnist_dec/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeEmnist_dec/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Dmnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
29
7mnist_dec/separable_conv2d_3/separable_conv2d/depthwiseÕ
-mnist_dec/separable_conv2d_3/separable_conv2dConv2D@mnist_dec/separable_conv2d_3/separable_conv2d/depthwise:output:0Fmnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_dec/separable_conv2d_3/separable_conv2dã
3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp<mnist_dec_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp
$mnist_dec/separable_conv2d_3/BiasAddBiasAdd6mnist_dec/separable_conv2d_3/separable_conv2d:output:0;mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_dec/separable_conv2d_3/BiasAdd·
!mnist_dec/separable_conv2d_3/SeluSelu-mnist_dec/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_dec/separable_conv2d_3/Selu
mnist_dec/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2!
mnist_dec/up_sampling2d_1/Const
!mnist_dec/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!mnist_dec/up_sampling2d_1/Const_1À
mnist_dec/up_sampling2d_1/mulMul(mnist_dec/up_sampling2d_1/Const:output:0*mnist_dec/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
mnist_dec/up_sampling2d_1/mul±
6mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor/mnist_dec/separable_conv2d_3/Selu:activations:0!mnist_dec/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(28
6mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighbor
<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpEmnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp
>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ã
3mnist_dec/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_dec/separable_conv2d_4/separable_conv2d/ShapeË
;mnist_dec/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_dec/separable_conv2d_4/separable_conv2d/dilation_rateü
7mnist_dec/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeGmnist_dec/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Dmnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
29
7mnist_dec/separable_conv2d_4/separable_conv2d/depthwiseÕ
-mnist_dec/separable_conv2d_4/separable_conv2dConv2D@mnist_dec/separable_conv2d_4/separable_conv2d/depthwise:output:0Fmnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_dec/separable_conv2d_4/separable_conv2dã
3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp<mnist_dec_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp
$mnist_dec/separable_conv2d_4/BiasAddBiasAdd6mnist_dec/separable_conv2d_4/separable_conv2d:output:0;mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_dec/separable_conv2d_4/BiasAdd·
!mnist_dec/separable_conv2d_4/SeluSelu-mnist_dec/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_dec/separable_conv2d_4/SeluÎ
(mnist_dec/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1mnist_dec_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02*
(mnist_dec/conv2d_2/Conv2D/ReadVariableOp
mnist_dec/conv2d_2/Conv2DConv2D/mnist_dec/separable_conv2d_4/Selu:activations:00mnist_dec/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_dec/conv2d_2/Conv2DÅ
)mnist_dec/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2mnist_dec_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)mnist_dec/conv2d_2/BiasAdd/ReadVariableOpÔ
mnist_dec/conv2d_2/BiasAddBiasAdd"mnist_dec/conv2d_2/Conv2D:output:01mnist_dec/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec/conv2d_2/BiasAdd
mnist_dec/conv2d_2/SeluSelu#mnist_dec/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec/conv2d_2/Selu
IdentityIdentity%mnist_dec/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity×	
NoOpNoOp*^mnist_dec/conv2d_2/BiasAdd/ReadVariableOp)^mnist_dec/conv2d_2/Conv2D/ReadVariableOp4^mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp=^mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp?^mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_14^mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp=^mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp?^mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_14^mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp=^mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp?^mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1(^mnist_enc/conv2d/BiasAdd/ReadVariableOp'^mnist_enc/conv2d/Conv2D/ReadVariableOp*^mnist_enc/conv2d_1/BiasAdd/ReadVariableOp)^mnist_enc/conv2d_1/Conv2D/ReadVariableOp2^mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp;^mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp=^mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_14^mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp=^mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp?^mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2V
)mnist_dec/conv2d_2/BiasAdd/ReadVariableOp)mnist_dec/conv2d_2/BiasAdd/ReadVariableOp2T
(mnist_dec/conv2d_2/Conv2D/ReadVariableOp(mnist_dec/conv2d_2/Conv2D/ReadVariableOp2j
3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp2|
<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp2
>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_12j
3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp2|
<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp2
>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_12j
3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp2|
<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp2
>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_12R
'mnist_enc/conv2d/BiasAdd/ReadVariableOp'mnist_enc/conv2d/BiasAdd/ReadVariableOp2P
&mnist_enc/conv2d/Conv2D/ReadVariableOp&mnist_enc/conv2d/Conv2D/ReadVariableOp2V
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp2T
(mnist_enc/conv2d_1/Conv2D/ReadVariableOp(mnist_enc/conv2d_1/Conv2D/ReadVariableOp2f
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp2x
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp2|
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_12j
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp2|
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp2
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

(__inference_conv2d_2_layer_call_fn_97293

inputs!
unknown:1
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_959232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ô
Ñ
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96203

inputs)
mnist_enc_96158:
mnist_enc_96160:)
mnist_enc_96162:)
mnist_enc_96164:1
mnist_enc_96166:1)
mnist_enc_96168:1)
mnist_enc_96170:11
mnist_enc_96172:1)
mnist_enc_96174:11
mnist_enc_96176:1)
mnist_dec_96179:)
mnist_dec_96181:1
mnist_dec_96183:1)
mnist_dec_96185:1)
mnist_dec_96187:11
mnist_dec_96189:1)
mnist_dec_96191:1)
mnist_dec_96193:11
mnist_dec_96195:1)
mnist_dec_96197:1
mnist_dec_96199:
identity¢!mnist_dec/StatefulPartitionedCall¢!mnist_enc/StatefulPartitionedCall±
!mnist_enc/StatefulPartitionedCallStatefulPartitionedCallinputsmnist_enc_96158mnist_enc_96160mnist_enc_96162mnist_enc_96164mnist_enc_96166mnist_enc_96168mnist_enc_96170mnist_enc_96172mnist_enc_96174mnist_enc_96176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_955362#
!mnist_enc/StatefulPartitionedCall
!mnist_dec/StatefulPartitionedCallStatefulPartitionedCall*mnist_enc/StatefulPartitionedCall:output:0mnist_dec_96179mnist_dec_96181mnist_dec_96183mnist_dec_96185mnist_dec_96187mnist_dec_96189mnist_dec_96191mnist_dec_96193mnist_dec_96195mnist_dec_96197mnist_dec_96199*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_959302#
!mnist_dec/StatefulPartitionedCall
IdentityIdentity*mnist_dec/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^mnist_dec/StatefulPartitionedCall"^mnist_enc/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2F
!mnist_dec/StatefulPartitionedCall!mnist_dec/StatefulPartitionedCall2F
!mnist_enc/StatefulPartitionedCall!mnist_enc/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·'

D__inference_mnist_dec_layer_call_and_return_conditional_losses_96151

dec_in2
separable_conv2d_2_96122:2
separable_conv2d_2_96124:1&
separable_conv2d_2_96126:12
separable_conv2d_3_96130:12
separable_conv2d_3_96132:11&
separable_conv2d_3_96134:12
separable_conv2d_4_96138:12
separable_conv2d_4_96140:11&
separable_conv2d_4_96142:1(
conv2d_2_96145:1
conv2d_2_96147:
identity¢ conv2d_2/StatefulPartitionedCall¢*separable_conv2d_2/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCallÛ
reshape/PartitionedCallPartitionedCalldec_in*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_958872
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96122separable_conv2d_2_96124separable_conv2d_2_96126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_957582,
*separable_conv2d_2/StatefulPartitionedCall¬
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_957832
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96130separable_conv2d_3_96132separable_conv2d_3_96134*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_958062,
*separable_conv2d_3/StatefulPartitionedCall²
up_sampling2d_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_958312!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96138separable_conv2d_4_96140separable_conv2d_4_96142*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_958542,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96145conv2d_2_96147*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_959232"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityø
NoOpNoOp!^conv2d_2/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_namedec_in
Ô
Ñ
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96485

enc_in)
mnist_enc_96440:
mnist_enc_96442:)
mnist_enc_96444:)
mnist_enc_96446:1
mnist_enc_96448:1)
mnist_enc_96450:1)
mnist_enc_96452:11
mnist_enc_96454:1)
mnist_enc_96456:11
mnist_enc_96458:1)
mnist_dec_96461:)
mnist_dec_96463:1
mnist_dec_96465:1)
mnist_dec_96467:1)
mnist_dec_96469:11
mnist_dec_96471:1)
mnist_dec_96473:1)
mnist_dec_96475:11
mnist_dec_96477:1)
mnist_dec_96479:1
mnist_dec_96481:
identity¢!mnist_dec/StatefulPartitionedCall¢!mnist_enc/StatefulPartitionedCall±
!mnist_enc/StatefulPartitionedCallStatefulPartitionedCallenc_inmnist_enc_96440mnist_enc_96442mnist_enc_96444mnist_enc_96446mnist_enc_96448mnist_enc_96450mnist_enc_96452mnist_enc_96454mnist_enc_96456mnist_enc_96458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_955362#
!mnist_enc/StatefulPartitionedCall
!mnist_dec/StatefulPartitionedCallStatefulPartitionedCall*mnist_enc/StatefulPartitionedCall:output:0mnist_dec_96461mnist_dec_96463mnist_dec_96465mnist_dec_96467mnist_dec_96469mnist_dec_96471mnist_dec_96473mnist_dec_96475mnist_dec_96477mnist_dec_96479mnist_dec_96481*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_959302#
!mnist_dec/StatefulPartitionedCall
IdentityIdentity*mnist_dec/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^mnist_dec/StatefulPartitionedCall"^mnist_enc/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2F
!mnist_dec/StatefulPartitionedCall!mnist_dec/StatefulPartitionedCall2F
!mnist_enc/StatefulPartitionedCall!mnist_enc/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
È
Þ
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96798

inputsI
/mnist_enc_conv2d_conv2d_readvariableop_resource:>
0mnist_enc_conv2d_biasadd_readvariableop_resource:]
Cmnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource:_
Emnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource:1H
:mnist_enc_separable_conv2d_biasadd_readvariableop_resource:1_
Emnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource:1a
Gmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11J
<mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource:1K
1mnist_enc_conv2d_1_conv2d_readvariableop_resource:11@
2mnist_enc_conv2d_1_biasadd_readvariableop_resource:1_
Emnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_resource:a
Gmnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1J
<mnist_dec_separable_conv2d_2_biasadd_readvariableop_resource:1_
Emnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_resource:1a
Gmnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11J
<mnist_dec_separable_conv2d_3_biasadd_readvariableop_resource:1_
Emnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_resource:1a
Gmnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11J
<mnist_dec_separable_conv2d_4_biasadd_readvariableop_resource:1K
1mnist_dec_conv2d_2_conv2d_readvariableop_resource:1@
2mnist_dec_conv2d_2_biasadd_readvariableop_resource:
identity¢)mnist_dec/conv2d_2/BiasAdd/ReadVariableOp¢(mnist_dec/conv2d_2/Conv2D/ReadVariableOp¢3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp¢<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp¢>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp¢<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp¢>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp¢<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp¢>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1¢'mnist_enc/conv2d/BiasAdd/ReadVariableOp¢&mnist_enc/conv2d/Conv2D/ReadVariableOp¢)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp¢(mnist_enc/conv2d_1/Conv2D/ReadVariableOp¢1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp¢:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp¢<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1¢3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp¢<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp¢>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1È
&mnist_enc/conv2d/Conv2D/ReadVariableOpReadVariableOp/mnist_enc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&mnist_enc/conv2d/Conv2D/ReadVariableOpÖ
mnist_enc/conv2d/Conv2DConv2Dinputs.mnist_enc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_enc/conv2d/Conv2D¿
'mnist_enc/conv2d/BiasAdd/ReadVariableOpReadVariableOp0mnist_enc_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'mnist_enc/conv2d/BiasAdd/ReadVariableOpÌ
mnist_enc/conv2d/BiasAddBiasAdd mnist_enc/conv2d/Conv2D:output:0/mnist_enc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_enc/conv2d/BiasAdd
mnist_enc/conv2d/SeluSelu!mnist_enc/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_enc/conv2d/Selu
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpCmnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02<
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpEmnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02>
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1¿
1mnist_enc/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1mnist_enc/separable_conv2d/separable_conv2d/ShapeÇ
9mnist_enc/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2;
9mnist_enc/separable_conv2d/separable_conv2d/dilation_rateÒ
5mnist_enc/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative#mnist_enc/conv2d/Selu:activations:0Bmnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
27
5mnist_enc/separable_conv2d/separable_conv2d/depthwiseÍ
+mnist_enc/separable_conv2d/separable_conv2dConv2D>mnist_enc/separable_conv2d/separable_conv2d/depthwise:output:0Dmnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2-
+mnist_enc/separable_conv2d/separable_conv2dÝ
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp:mnist_enc_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype023
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpþ
"mnist_enc/separable_conv2d/BiasAddBiasAdd4mnist_enc/separable_conv2d/separable_conv2d:output:09mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12$
"mnist_enc/separable_conv2d/BiasAdd±
mnist_enc/separable_conv2d/SeluSelu+mnist_enc/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12!
mnist_enc/separable_conv2d/Selu
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpEmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ã
3mnist_enc/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_enc/separable_conv2d_1/separable_conv2d/ShapeË
;mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rateâ
7mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative-mnist_enc/separable_conv2d/Selu:activations:0Dmnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
29
7mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseÕ
-mnist_enc/separable_conv2d_1/separable_conv2dConv2D@mnist_enc/separable_conv2d_1/separable_conv2d/depthwise:output:0Fmnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_enc/separable_conv2d_1/separable_conv2dã
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp
$mnist_enc/separable_conv2d_1/BiasAddBiasAdd6mnist_enc/separable_conv2d_1/separable_conv2d:output:0;mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_enc/separable_conv2d_1/BiasAdd·
!mnist_enc/separable_conv2d_1/SeluSelu-mnist_enc/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_enc/separable_conv2d_1/SeluÎ
(mnist_enc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1mnist_enc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02*
(mnist_enc/conv2d_1/Conv2D/ReadVariableOp
mnist_enc/conv2d_1/Conv2DConv2D/mnist_enc/separable_conv2d_1/Selu:activations:00mnist_enc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
mnist_enc/conv2d_1/Conv2DÅ
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2mnist_enc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOpÔ
mnist_enc/conv2d_1/BiasAddBiasAdd"mnist_enc/conv2d_1/Conv2D:output:01mnist_enc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc/conv2d_1/BiasAdd
mnist_enc/conv2d_1/SeluSelu#mnist_enc/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc/conv2d_1/SeluÇ
9mnist_enc/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9mnist_enc/global_average_pooling2d/Mean/reduction_indices÷
'mnist_enc/global_average_pooling2d/MeanMean%mnist_enc/conv2d_1/Selu:activations:0Bmnist_enc/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_enc/global_average_pooling2d/Mean
mnist_dec/reshape/ShapeShape0mnist_enc/global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:2
mnist_dec/reshape/Shape
%mnist_dec/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%mnist_dec/reshape/strided_slice/stack
'mnist_dec/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'mnist_dec/reshape/strided_slice/stack_1
'mnist_dec/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'mnist_dec/reshape/strided_slice/stack_2Î
mnist_dec/reshape/strided_sliceStridedSlice mnist_dec/reshape/Shape:output:0.mnist_dec/reshape/strided_slice/stack:output:00mnist_dec/reshape/strided_slice/stack_1:output:00mnist_dec/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
mnist_dec/reshape/strided_slice
!mnist_dec/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!mnist_dec/reshape/Reshape/shape/1
!mnist_dec/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!mnist_dec/reshape/Reshape/shape/2
!mnist_dec/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!mnist_dec/reshape/Reshape/shape/3¦
mnist_dec/reshape/Reshape/shapePack(mnist_dec/reshape/strided_slice:output:0*mnist_dec/reshape/Reshape/shape/1:output:0*mnist_dec/reshape/Reshape/shape/2:output:0*mnist_dec/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
mnist_dec/reshape/Reshape/shape×
mnist_dec/reshape/ReshapeReshape0mnist_enc/global_average_pooling2d/Mean:output:0(mnist_dec/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec/reshape/Reshape
<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpEmnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp
>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02@
>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ã
3mnist_dec/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            25
3mnist_dec/separable_conv2d_2/separable_conv2d/ShapeË
;mnist_dec/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_dec/separable_conv2d_2/separable_conv2d/dilation_rate×
7mnist_dec/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative"mnist_dec/reshape/Reshape:output:0Dmnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
29
7mnist_dec/separable_conv2d_2/separable_conv2d/depthwiseÕ
-mnist_dec/separable_conv2d_2/separable_conv2dConv2D@mnist_dec/separable_conv2d_2/separable_conv2d/depthwise:output:0Fmnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_dec/separable_conv2d_2/separable_conv2dã
3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<mnist_dec_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp
$mnist_dec/separable_conv2d_2/BiasAddBiasAdd6mnist_dec/separable_conv2d_2/separable_conv2d:output:0;mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_dec/separable_conv2d_2/BiasAdd·
!mnist_dec/separable_conv2d_2/SeluSelu-mnist_dec/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_dec/separable_conv2d_2/Selu
mnist_dec/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
mnist_dec/up_sampling2d/Const
mnist_dec/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2!
mnist_dec/up_sampling2d/Const_1¸
mnist_dec/up_sampling2d/mulMul&mnist_dec/up_sampling2d/Const:output:0(mnist_dec/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
mnist_dec/up_sampling2d/mul«
4mnist_dec/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor/mnist_dec/separable_conv2d_2/Selu:activations:0mnist_dec/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(26
4mnist_dec/up_sampling2d/resize/ResizeNearestNeighbor
<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpEmnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp
>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ã
3mnist_dec/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_dec/separable_conv2d_3/separable_conv2d/ShapeË
;mnist_dec/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_dec/separable_conv2d_3/separable_conv2d/dilation_rateú
7mnist_dec/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeEmnist_dec/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Dmnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
29
7mnist_dec/separable_conv2d_3/separable_conv2d/depthwiseÕ
-mnist_dec/separable_conv2d_3/separable_conv2dConv2D@mnist_dec/separable_conv2d_3/separable_conv2d/depthwise:output:0Fmnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_dec/separable_conv2d_3/separable_conv2dã
3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp<mnist_dec_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp
$mnist_dec/separable_conv2d_3/BiasAddBiasAdd6mnist_dec/separable_conv2d_3/separable_conv2d:output:0;mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_dec/separable_conv2d_3/BiasAdd·
!mnist_dec/separable_conv2d_3/SeluSelu-mnist_dec/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_dec/separable_conv2d_3/Selu
mnist_dec/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2!
mnist_dec/up_sampling2d_1/Const
!mnist_dec/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!mnist_dec/up_sampling2d_1/Const_1À
mnist_dec/up_sampling2d_1/mulMul(mnist_dec/up_sampling2d_1/Const:output:0*mnist_dec/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
mnist_dec/up_sampling2d_1/mul±
6mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor/mnist_dec/separable_conv2d_3/Selu:activations:0!mnist_dec/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(28
6mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighbor
<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpEmnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02>
<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp
>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpGmnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02@
>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ã
3mnist_dec/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      25
3mnist_dec/separable_conv2d_4/separable_conv2d/ShapeË
;mnist_dec/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;mnist_dec/separable_conv2d_4/separable_conv2d/dilation_rateü
7mnist_dec/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeGmnist_dec/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Dmnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
29
7mnist_dec/separable_conv2d_4/separable_conv2d/depthwiseÕ
-mnist_dec/separable_conv2d_4/separable_conv2dConv2D@mnist_dec/separable_conv2d_4/separable_conv2d/depthwise:output:0Fmnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2/
-mnist_dec/separable_conv2d_4/separable_conv2dã
3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp<mnist_dec_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype025
3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp
$mnist_dec/separable_conv2d_4/BiasAddBiasAdd6mnist_dec/separable_conv2d_4/separable_conv2d:output:0;mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_dec/separable_conv2d_4/BiasAdd·
!mnist_dec/separable_conv2d_4/SeluSelu-mnist_dec/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12#
!mnist_dec/separable_conv2d_4/SeluÎ
(mnist_dec/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1mnist_dec_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02*
(mnist_dec/conv2d_2/Conv2D/ReadVariableOp
mnist_dec/conv2d_2/Conv2DConv2D/mnist_dec/separable_conv2d_4/Selu:activations:00mnist_dec/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_dec/conv2d_2/Conv2DÅ
)mnist_dec/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2mnist_dec_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)mnist_dec/conv2d_2/BiasAdd/ReadVariableOpÔ
mnist_dec/conv2d_2/BiasAddBiasAdd"mnist_dec/conv2d_2/Conv2D:output:01mnist_dec/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec/conv2d_2/BiasAdd
mnist_dec/conv2d_2/SeluSelu#mnist_dec/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec/conv2d_2/Selu
IdentityIdentity%mnist_dec/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity×	
NoOpNoOp*^mnist_dec/conv2d_2/BiasAdd/ReadVariableOp)^mnist_dec/conv2d_2/Conv2D/ReadVariableOp4^mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp=^mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp?^mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_14^mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp=^mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp?^mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_14^mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp=^mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp?^mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1(^mnist_enc/conv2d/BiasAdd/ReadVariableOp'^mnist_enc/conv2d/Conv2D/ReadVariableOp*^mnist_enc/conv2d_1/BiasAdd/ReadVariableOp)^mnist_enc/conv2d_1/Conv2D/ReadVariableOp2^mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp;^mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp=^mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_14^mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp=^mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp?^mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2V
)mnist_dec/conv2d_2/BiasAdd/ReadVariableOp)mnist_dec/conv2d_2/BiasAdd/ReadVariableOp2T
(mnist_dec/conv2d_2/Conv2D/ReadVariableOp(mnist_dec/conv2d_2/Conv2D/ReadVariableOp2j
3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp3mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp2|
<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp<mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp2
>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1>mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_12j
3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp3mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp2|
<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp<mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp2
>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1>mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_12j
3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp3mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp2|
<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp<mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp2
>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1>mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_12R
'mnist_enc/conv2d/BiasAdd/ReadVariableOp'mnist_enc/conv2d/BiasAdd/ReadVariableOp2P
&mnist_enc/conv2d/Conv2D/ReadVariableOp&mnist_enc/conv2d/Conv2D/ReadVariableOp2V
)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp)mnist_enc/conv2d_1/BiasAdd/ReadVariableOp2T
(mnist_enc/conv2d_1/Conv2D/ReadVariableOp(mnist_enc/conv2d_1/Conv2D/ReadVariableOp2f
1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp1mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp2x
:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp2|
<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1<mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_12j
3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp3mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp2|
<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp<mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp2
>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1>mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

(__inference_mnist_ae_layer_call_fn_96248

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
	unknown_8:1#
	unknown_9:$

unknown_10:1

unknown_11:1$

unknown_12:1$

unknown_13:11

unknown_14:1$

unknown_15:1$

unknown_16:11

unknown_17:1$

unknown_18:1

unknown_19:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_mnist_ae_layer_call_and_return_conditional_losses_962032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
ç
ú
A__inference_conv2d_layer_call_and_return_conditional_losses_95497

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_95783

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ðh
¿
D__inference_mnist_dec_layer_call_and_return_conditional_losses_97160

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
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢)separable_conv2d_2/BiasAdd/ReadVariableOp¢2separable_conv2d_2/separable_conv2d/ReadVariableOp¢4separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_3/BiasAdd/ReadVariableOp¢2separable_conv2d_3/separable_conv2d/ReadVariableOp¢4separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_4/BiasAdd/ReadVariableOp¢2separable_conv2d_4/separable_conv2d/ReadVariableOp¢4separable_conv2d_4/separable_conv2d/ReadVariableOp_1T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shape/3ê
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshapeì
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpò
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_2/separable_conv2d/Shape·
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rate¯
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativereshape/Reshape:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwise­
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2dÅ
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOpÞ
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_2/BiasAdd
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
up_sampling2d/Const_1
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_2/Selu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborì
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpò
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_3/separable_conv2d/Shape·
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rateÒ
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise­
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2dÅ
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpÞ
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_3/BiasAdd
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_3/Selu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_3/Selu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborì
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpò
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2+
)separable_conv2d_4/separable_conv2d/Shape·
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rateÔ
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise­
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2dÅ
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpÞ
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_4/BiasAdd
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
separable_conv2d_4/Selu°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÝ
conv2d_2/Conv2DConv2D%separable_conv2d_4/Selu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd{
conv2d_2/SeluSeluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Selu~
IdentityIdentityconv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÙ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 2B
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
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é
ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97245

inputs8
conv2d_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¦

M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95454

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identity­
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
ú
A__inference_conv2d_layer_call_and_return_conditional_losses_97225

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Seluu
IdentityIdentitySelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_conv2d_1_layer_call_fn_97254

inputs!
unknown:11
	unknown_0:1
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_955282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
«	
Ì
2__inference_separable_conv2d_2_layer_call_fn_95770

inputs!
unknown:#
	unknown_0:1
	unknown_1:1
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_957582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
Ç
)__inference_mnist_dec_layer_call_fn_97187

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_959302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ1: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
·
3
!__inference__traced_restore_97746
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: :
 assignvariableop_5_conv2d_kernel:,
assignvariableop_6_conv2d_bias:N
4assignvariableop_7_separable_conv2d_depthwise_kernel:N
4assignvariableop_8_separable_conv2d_pointwise_kernel:16
(assignvariableop_9_separable_conv2d_bias:1Q
7assignvariableop_10_separable_conv2d_1_depthwise_kernel:1Q
7assignvariableop_11_separable_conv2d_1_pointwise_kernel:119
+assignvariableop_12_separable_conv2d_1_bias:1=
#assignvariableop_13_conv2d_1_kernel:11/
!assignvariableop_14_conv2d_1_bias:1Q
7assignvariableop_15_separable_conv2d_2_depthwise_kernel:Q
7assignvariableop_16_separable_conv2d_2_pointwise_kernel:19
+assignvariableop_17_separable_conv2d_2_bias:1Q
7assignvariableop_18_separable_conv2d_3_depthwise_kernel:1Q
7assignvariableop_19_separable_conv2d_3_pointwise_kernel:119
+assignvariableop_20_separable_conv2d_3_bias:1Q
7assignvariableop_21_separable_conv2d_4_depthwise_kernel:1Q
7assignvariableop_22_separable_conv2d_4_pointwise_kernel:119
+assignvariableop_23_separable_conv2d_4_bias:1=
#assignvariableop_24_conv2d_2_kernel:1/
!assignvariableop_25_conv2d_2_bias:#
assignvariableop_26_total: #
assignvariableop_27_count: B
(assignvariableop_28_adam_conv2d_kernel_m:4
&assignvariableop_29_adam_conv2d_bias_m:V
<assignvariableop_30_adam_separable_conv2d_depthwise_kernel_m:V
<assignvariableop_31_adam_separable_conv2d_pointwise_kernel_m:1>
0assignvariableop_32_adam_separable_conv2d_bias_m:1X
>assignvariableop_33_adam_separable_conv2d_1_depthwise_kernel_m:1X
>assignvariableop_34_adam_separable_conv2d_1_pointwise_kernel_m:11@
2assignvariableop_35_adam_separable_conv2d_1_bias_m:1D
*assignvariableop_36_adam_conv2d_1_kernel_m:116
(assignvariableop_37_adam_conv2d_1_bias_m:1X
>assignvariableop_38_adam_separable_conv2d_2_depthwise_kernel_m:X
>assignvariableop_39_adam_separable_conv2d_2_pointwise_kernel_m:1@
2assignvariableop_40_adam_separable_conv2d_2_bias_m:1X
>assignvariableop_41_adam_separable_conv2d_3_depthwise_kernel_m:1X
>assignvariableop_42_adam_separable_conv2d_3_pointwise_kernel_m:11@
2assignvariableop_43_adam_separable_conv2d_3_bias_m:1X
>assignvariableop_44_adam_separable_conv2d_4_depthwise_kernel_m:1X
>assignvariableop_45_adam_separable_conv2d_4_pointwise_kernel_m:11@
2assignvariableop_46_adam_separable_conv2d_4_bias_m:1D
*assignvariableop_47_adam_conv2d_2_kernel_m:16
(assignvariableop_48_adam_conv2d_2_bias_m:B
(assignvariableop_49_adam_conv2d_kernel_v:4
&assignvariableop_50_adam_conv2d_bias_v:V
<assignvariableop_51_adam_separable_conv2d_depthwise_kernel_v:V
<assignvariableop_52_adam_separable_conv2d_pointwise_kernel_v:1>
0assignvariableop_53_adam_separable_conv2d_bias_v:1X
>assignvariableop_54_adam_separable_conv2d_1_depthwise_kernel_v:1X
>assignvariableop_55_adam_separable_conv2d_1_pointwise_kernel_v:11@
2assignvariableop_56_adam_separable_conv2d_1_bias_v:1D
*assignvariableop_57_adam_conv2d_1_kernel_v:116
(assignvariableop_58_adam_conv2d_1_bias_v:1X
>assignvariableop_59_adam_separable_conv2d_2_depthwise_kernel_v:X
>assignvariableop_60_adam_separable_conv2d_2_pointwise_kernel_v:1@
2assignvariableop_61_adam_separable_conv2d_2_bias_v:1X
>assignvariableop_62_adam_separable_conv2d_3_depthwise_kernel_v:1X
>assignvariableop_63_adam_separable_conv2d_3_pointwise_kernel_v:11@
2assignvariableop_64_adam_separable_conv2d_3_bias_v:1X
>assignvariableop_65_adam_separable_conv2d_4_depthwise_kernel_v:1X
>assignvariableop_66_adam_separable_conv2d_4_pointwise_kernel_v:11@
2assignvariableop_67_adam_separable_conv2d_4_bias_v:1D
*assignvariableop_68_adam_conv2d_2_kernel_v:16
(assignvariableop_69_adam_conv2d_2_bias_v:
identity_71¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ÿ%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*%
value%Bþ$GB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*£
valueBGB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*²
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¹
AssignVariableOp_7AssignVariableOp4assignvariableop_7_separable_conv2d_depthwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¹
AssignVariableOp_8AssignVariableOp4assignvariableop_8_separable_conv2d_pointwise_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_separable_conv2d_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¿
AssignVariableOp_10AssignVariableOp7assignvariableop_10_separable_conv2d_1_depthwise_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¿
AssignVariableOp_11AssignVariableOp7assignvariableop_11_separable_conv2d_1_pointwise_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12³
AssignVariableOp_12AssignVariableOp+assignvariableop_12_separable_conv2d_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13«
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¿
AssignVariableOp_15AssignVariableOp7assignvariableop_15_separable_conv2d_2_depthwise_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¿
AssignVariableOp_16AssignVariableOp7assignvariableop_16_separable_conv2d_2_pointwise_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17³
AssignVariableOp_17AssignVariableOp+assignvariableop_17_separable_conv2d_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¿
AssignVariableOp_18AssignVariableOp7assignvariableop_18_separable_conv2d_3_depthwise_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¿
AssignVariableOp_19AssignVariableOp7assignvariableop_19_separable_conv2d_3_pointwise_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20³
AssignVariableOp_20AssignVariableOp+assignvariableop_20_separable_conv2d_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¿
AssignVariableOp_21AssignVariableOp7assignvariableop_21_separable_conv2d_4_depthwise_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¿
AssignVariableOp_22AssignVariableOp7assignvariableop_22_separable_conv2d_4_pointwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_separable_conv2d_4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25©
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¡
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¡
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29®
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_conv2d_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ä
AssignVariableOp_30AssignVariableOp<assignvariableop_30_adam_separable_conv2d_depthwise_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ä
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_separable_conv2d_pointwise_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¸
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_separable_conv2d_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Æ
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_separable_conv2d_1_depthwise_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Æ
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_separable_conv2d_1_pointwise_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35º
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_separable_conv2d_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36²
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_1_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_1_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Æ
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_separable_conv2d_2_depthwise_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Æ
AssignVariableOp_39AssignVariableOp>assignvariableop_39_adam_separable_conv2d_2_pointwise_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40º
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_separable_conv2d_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Æ
AssignVariableOp_41AssignVariableOp>assignvariableop_41_adam_separable_conv2d_3_depthwise_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Æ
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_separable_conv2d_3_pointwise_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43º
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_separable_conv2d_3_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Æ
AssignVariableOp_44AssignVariableOp>assignvariableop_44_adam_separable_conv2d_4_depthwise_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Æ
AssignVariableOp_45AssignVariableOp>assignvariableop_45_adam_separable_conv2d_4_pointwise_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46º
AssignVariableOp_46AssignVariableOp2assignvariableop_46_adam_separable_conv2d_4_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49°
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv2d_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50®
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_conv2d_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ä
AssignVariableOp_51AssignVariableOp<assignvariableop_51_adam_separable_conv2d_depthwise_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ä
AssignVariableOp_52AssignVariableOp<assignvariableop_52_adam_separable_conv2d_pointwise_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¸
AssignVariableOp_53AssignVariableOp0assignvariableop_53_adam_separable_conv2d_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Æ
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_separable_conv2d_1_depthwise_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Æ
AssignVariableOp_55AssignVariableOp>assignvariableop_55_adam_separable_conv2d_1_pointwise_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56º
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_separable_conv2d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57²
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv2d_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58°
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Æ
AssignVariableOp_59AssignVariableOp>assignvariableop_59_adam_separable_conv2d_2_depthwise_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Æ
AssignVariableOp_60AssignVariableOp>assignvariableop_60_adam_separable_conv2d_2_pointwise_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61º
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_separable_conv2d_2_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Æ
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_separable_conv2d_3_depthwise_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Æ
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adam_separable_conv2d_3_pointwise_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64º
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_separable_conv2d_3_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Æ
AssignVariableOp_65AssignVariableOp>assignvariableop_65_adam_separable_conv2d_4_depthwise_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Æ
AssignVariableOp_66AssignVariableOp>assignvariableop_66_adam_separable_conv2d_4_pointwise_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67º
AssignVariableOp_67AssignVariableOp2assignvariableop_67_adam_separable_conv2d_4_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68²
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_conv2d_2_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69°
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_conv2d_2_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_699
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpâ
Identity_70Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_70f
Identity_71IdentityIdentity_70:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_71Ê
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_71Identity_71:output:0*£
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤

K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95425

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ12

Identity­
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


&__inference_conv2d_layer_call_fn_97234

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_954972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
¥
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95536

inputs&
conv2d_95498:
conv2d_95500:0
separable_conv2d_95503:0
separable_conv2d_95505:1$
separable_conv2d_95507:12
separable_conv2d_1_95510:12
separable_conv2d_1_95512:11&
separable_conv2d_1_95514:1(
conv2d_1_95529:11
conv2d_1_95531:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_95498conv2d_95500*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_954972 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_95503separable_conv2d_95505separable_conv2d_95507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_954252*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95510separable_conv2d_1_95512separable_conv2d_1_95514*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_954542,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_95529conv2d_1_95531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_955282"
 conv2d_1/StatefulPartitionedCall©
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_954732*
(global_average_pooling2d/PartitionedCall
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityê
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
µ
 __inference__wrapped_model_95408

enc_inR
8mnist_ae_mnist_enc_conv2d_conv2d_readvariableop_resource:G
9mnist_ae_mnist_enc_conv2d_biasadd_readvariableop_resource:f
Lmnist_ae_mnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource:h
Nmnist_ae_mnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource:1Q
Cmnist_ae_mnist_enc_separable_conv2d_biasadd_readvariableop_resource:1h
Nmnist_ae_mnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource:1j
Pmnist_ae_mnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11S
Emnist_ae_mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource:1T
:mnist_ae_mnist_enc_conv2d_1_conv2d_readvariableop_resource:11I
;mnist_ae_mnist_enc_conv2d_1_biasadd_readvariableop_resource:1h
Nmnist_ae_mnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_resource:j
Pmnist_ae_mnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1S
Emnist_ae_mnist_dec_separable_conv2d_2_biasadd_readvariableop_resource:1h
Nmnist_ae_mnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_resource:1j
Pmnist_ae_mnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11S
Emnist_ae_mnist_dec_separable_conv2d_3_biasadd_readvariableop_resource:1h
Nmnist_ae_mnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_resource:1j
Pmnist_ae_mnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11S
Emnist_ae_mnist_dec_separable_conv2d_4_biasadd_readvariableop_resource:1T
:mnist_ae_mnist_dec_conv2d_2_conv2d_readvariableop_resource:1I
;mnist_ae_mnist_dec_conv2d_2_biasadd_readvariableop_resource:
identity¢2mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOp¢1mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOp¢<mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp¢Emnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp¢Gmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢<mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp¢Emnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp¢Gmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢<mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp¢Emnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp¢Gmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1¢0mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOp¢/mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOp¢2mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOp¢1mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOp¢:mnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp¢Cmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp¢Emnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1¢<mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp¢Emnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp¢Gmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ã
/mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOpReadVariableOp8mnist_ae_mnist_enc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOpñ
 mnist_ae/mnist_enc/conv2d/Conv2DConv2Denc_in7mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 mnist_ae/mnist_enc/conv2d/Conv2DÚ
0mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOpReadVariableOp9mnist_ae_mnist_enc_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOpð
!mnist_ae/mnist_enc/conv2d/BiasAddBiasAdd)mnist_ae/mnist_enc/conv2d/Conv2D:output:08mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!mnist_ae/mnist_enc/conv2d/BiasAdd®
mnist_ae/mnist_enc/conv2d/SeluSelu*mnist_ae/mnist_enc/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
mnist_ae/mnist_enc/conv2d/Selu
Cmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpLmnist_ae_mnist_enc_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02E
Cmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp¥
Emnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpNmnist_ae_mnist_enc_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02G
Emnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1Ñ
:mnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2<
:mnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ShapeÙ
Bmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/dilation_rateö
>mnist_ae/mnist_enc/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative,mnist_ae/mnist_enc/conv2d/Selu:activations:0Kmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2@
>mnist_ae/mnist_enc/separable_conv2d/separable_conv2d/depthwiseñ
4mnist_ae/mnist_enc/separable_conv2d/separable_conv2dConv2DGmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/depthwise:output:0Mmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
26
4mnist_ae/mnist_enc/separable_conv2d/separable_conv2dø
:mnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOpCmnist_ae_mnist_enc_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02<
:mnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp¢
+mnist_ae/mnist_enc/separable_conv2d/BiasAddBiasAdd=mnist_ae/mnist_enc/separable_conv2d/separable_conv2d:output:0Bmnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12-
+mnist_ae/mnist_enc/separable_conv2d/BiasAddÌ
(mnist_ae/mnist_enc/separable_conv2d/SeluSelu4mnist_ae/mnist_enc/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_ae/mnist_enc/separable_conv2d/Selu¥
Emnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpNmnist_ae_mnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02G
Emnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp«
Gmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpPmnist_ae_mnist_enc_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02I
Gmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Õ
<mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2>
<mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ShapeÝ
Dmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2F
Dmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/dilation_rate
@mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative6mnist_ae/mnist_enc/separable_conv2d/Selu:activations:0Mmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2B
@mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/depthwiseù
6mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2dConv2DImnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/depthwise:output:0Omnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
28
6mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2dþ
<mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpEmnist_ae_mnist_enc_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02>
<mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOpª
-mnist_ae/mnist_enc/separable_conv2d_1/BiasAddBiasAdd?mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d:output:0Dmnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12/
-mnist_ae/mnist_enc/separable_conv2d_1/BiasAddÒ
*mnist_ae/mnist_enc/separable_conv2d_1/SeluSelu6mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12,
*mnist_ae/mnist_enc/separable_conv2d_1/Selué
1mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:mnist_ae_mnist_enc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype023
1mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOp©
"mnist_ae/mnist_enc/conv2d_1/Conv2DConv2D8mnist_ae/mnist_enc/separable_conv2d_1/Selu:activations:09mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2$
"mnist_ae/mnist_enc/conv2d_1/Conv2Dà
2mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;mnist_ae_mnist_enc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype024
2mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOpø
#mnist_ae/mnist_enc/conv2d_1/BiasAddBiasAdd+mnist_ae/mnist_enc/conv2d_1/Conv2D:output:0:mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12%
#mnist_ae/mnist_enc/conv2d_1/BiasAdd´
 mnist_ae/mnist_enc/conv2d_1/SeluSelu,mnist_ae/mnist_enc/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12"
 mnist_ae/mnist_enc/conv2d_1/SeluÙ
Bmnist_ae/mnist_enc/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bmnist_ae/mnist_enc/global_average_pooling2d/Mean/reduction_indices
0mnist_ae/mnist_enc/global_average_pooling2d/MeanMean.mnist_ae/mnist_enc/conv2d_1/Selu:activations:0Kmnist_ae/mnist_enc/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ122
0mnist_ae/mnist_enc/global_average_pooling2d/Mean­
 mnist_ae/mnist_dec/reshape/ShapeShape9mnist_ae/mnist_enc/global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:2"
 mnist_ae/mnist_dec/reshape/Shapeª
.mnist_ae/mnist_dec/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.mnist_ae/mnist_dec/reshape/strided_slice/stack®
0mnist_ae/mnist_dec/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0mnist_ae/mnist_dec/reshape/strided_slice/stack_1®
0mnist_ae/mnist_dec/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0mnist_ae/mnist_dec/reshape/strided_slice/stack_2
(mnist_ae/mnist_dec/reshape/strided_sliceStridedSlice)mnist_ae/mnist_dec/reshape/Shape:output:07mnist_ae/mnist_dec/reshape/strided_slice/stack:output:09mnist_ae/mnist_dec/reshape/strided_slice/stack_1:output:09mnist_ae/mnist_dec/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(mnist_ae/mnist_dec/reshape/strided_slice
*mnist_ae/mnist_dec/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*mnist_ae/mnist_dec/reshape/Reshape/shape/1
*mnist_ae/mnist_dec/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2,
*mnist_ae/mnist_dec/reshape/Reshape/shape/2
*mnist_ae/mnist_dec/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2,
*mnist_ae/mnist_dec/reshape/Reshape/shape/3Ü
(mnist_ae/mnist_dec/reshape/Reshape/shapePack1mnist_ae/mnist_dec/reshape/strided_slice:output:03mnist_ae/mnist_dec/reshape/Reshape/shape/1:output:03mnist_ae/mnist_dec/reshape/Reshape/shape/2:output:03mnist_ae/mnist_dec/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2*
(mnist_ae/mnist_dec/reshape/Reshape/shapeû
"mnist_ae/mnist_dec/reshape/ReshapeReshape9mnist_ae/mnist_enc/global_average_pooling2d/Mean:output:01mnist_ae/mnist_dec/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"mnist_ae/mnist_dec/reshape/Reshape¥
Emnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpNmnist_ae_mnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02G
Emnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp«
Gmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpPmnist_ae_mnist_dec_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02I
Gmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Õ
<mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2>
<mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ShapeÝ
Dmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2F
Dmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/dilation_rateû
@mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative+mnist_ae/mnist_dec/reshape/Reshape:output:0Mmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2B
@mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/depthwiseù
6mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2dConv2DImnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/depthwise:output:0Omnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
28
6mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2dþ
<mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpEmnist_ae_mnist_dec_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02>
<mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOpª
-mnist_ae/mnist_dec/separable_conv2d_2/BiasAddBiasAdd?mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d:output:0Dmnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12/
-mnist_ae/mnist_dec/separable_conv2d_2/BiasAddÒ
*mnist_ae/mnist_dec/separable_conv2d_2/SeluSelu6mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12,
*mnist_ae/mnist_dec/separable_conv2d_2/Selu¡
&mnist_ae/mnist_dec/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2(
&mnist_ae/mnist_dec/up_sampling2d/Const¥
(mnist_ae/mnist_dec/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2*
(mnist_ae/mnist_dec/up_sampling2d/Const_1Ü
$mnist_ae/mnist_dec/up_sampling2d/mulMul/mnist_ae/mnist_dec/up_sampling2d/Const:output:01mnist_ae/mnist_dec/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2&
$mnist_ae/mnist_dec/up_sampling2d/mulÏ
=mnist_ae/mnist_dec/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor8mnist_ae/mnist_dec/separable_conv2d_2/Selu:activations:0(mnist_ae/mnist_dec/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2?
=mnist_ae/mnist_dec/up_sampling2d/resize/ResizeNearestNeighbor¥
Emnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpNmnist_ae_mnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02G
Emnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp«
Gmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpPmnist_ae_mnist_dec_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02I
Gmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Õ
<mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2>
<mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ShapeÝ
Dmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2F
Dmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/dilation_rate
@mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeNmnist_ae/mnist_dec/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Mmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2B
@mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/depthwiseù
6mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2dConv2DImnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/depthwise:output:0Omnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
28
6mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2dþ
<mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpEmnist_ae_mnist_dec_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02>
<mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOpª
-mnist_ae/mnist_dec/separable_conv2d_3/BiasAddBiasAdd?mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d:output:0Dmnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12/
-mnist_ae/mnist_dec/separable_conv2d_3/BiasAddÒ
*mnist_ae/mnist_dec/separable_conv2d_3/SeluSelu6mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12,
*mnist_ae/mnist_dec/separable_conv2d_3/Selu¥
(mnist_ae/mnist_dec/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2*
(mnist_ae/mnist_dec/up_sampling2d_1/Const©
*mnist_ae/mnist_dec/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2,
*mnist_ae/mnist_dec/up_sampling2d_1/Const_1ä
&mnist_ae/mnist_dec/up_sampling2d_1/mulMul1mnist_ae/mnist_dec/up_sampling2d_1/Const:output:03mnist_ae/mnist_dec/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2(
&mnist_ae/mnist_dec/up_sampling2d_1/mulÕ
?mnist_ae/mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor8mnist_ae/mnist_dec/separable_conv2d_3/Selu:activations:0*mnist_ae/mnist_dec/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2A
?mnist_ae/mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighbor¥
Emnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpNmnist_ae_mnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02G
Emnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp«
Gmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpPmnist_ae_mnist_dec_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02I
Gmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Õ
<mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2>
<mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ShapeÝ
Dmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2F
Dmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/dilation_rate 
@mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativePmnist_ae/mnist_dec/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Mmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2B
@mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/depthwiseù
6mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2dConv2DImnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/depthwise:output:0Omnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
28
6mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2dþ
<mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpEmnist_ae_mnist_dec_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02>
<mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOpª
-mnist_ae/mnist_dec/separable_conv2d_4/BiasAddBiasAdd?mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d:output:0Dmnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12/
-mnist_ae/mnist_dec/separable_conv2d_4/BiasAddÒ
*mnist_ae/mnist_dec/separable_conv2d_4/SeluSelu6mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12,
*mnist_ae/mnist_dec/separable_conv2d_4/Selué
1mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOpReadVariableOp:mnist_ae_mnist_dec_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype023
1mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOp©
"mnist_ae/mnist_dec/conv2d_2/Conv2DConv2D8mnist_ae/mnist_dec/separable_conv2d_4/Selu:activations:09mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2$
"mnist_ae/mnist_dec/conv2d_2/Conv2Dà
2mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp;mnist_ae_mnist_dec_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOpø
#mnist_ae/mnist_dec/conv2d_2/BiasAddBiasAdd+mnist_ae/mnist_dec/conv2d_2/Conv2D:output:0:mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#mnist_ae/mnist_dec/conv2d_2/BiasAdd´
 mnist_ae/mnist_dec/conv2d_2/SeluSelu,mnist_ae/mnist_dec/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 mnist_ae/mnist_dec/conv2d_2/Selu
IdentityIdentity.mnist_ae/mnist_dec/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp3^mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOp2^mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOp=^mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOpF^mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOpH^mnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1=^mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOpF^mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOpH^mnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1=^mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOpF^mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOpH^mnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_11^mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOp0^mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOp3^mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOp2^mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOp;^mnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOpD^mnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOpF^mnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1=^mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOpF^mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOpH^mnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2h
2mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOp2mnist_ae/mnist_dec/conv2d_2/BiasAdd/ReadVariableOp2f
1mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOp1mnist_ae/mnist_dec/conv2d_2/Conv2D/ReadVariableOp2|
<mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp<mnist_ae/mnist_dec/separable_conv2d_2/BiasAdd/ReadVariableOp2
Emnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOpEmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp2
Gmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Gmnist_ae/mnist_dec/separable_conv2d_2/separable_conv2d/ReadVariableOp_12|
<mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp<mnist_ae/mnist_dec/separable_conv2d_3/BiasAdd/ReadVariableOp2
Emnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOpEmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp2
Gmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Gmnist_ae/mnist_dec/separable_conv2d_3/separable_conv2d/ReadVariableOp_12|
<mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp<mnist_ae/mnist_dec/separable_conv2d_4/BiasAdd/ReadVariableOp2
Emnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOpEmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp2
Gmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Gmnist_ae/mnist_dec/separable_conv2d_4/separable_conv2d/ReadVariableOp_12d
0mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOp0mnist_ae/mnist_enc/conv2d/BiasAdd/ReadVariableOp2b
/mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOp/mnist_ae/mnist_enc/conv2d/Conv2D/ReadVariableOp2h
2mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOp2mnist_ae/mnist_enc/conv2d_1/BiasAdd/ReadVariableOp2f
1mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOp1mnist_ae/mnist_enc/conv2d_1/Conv2D/ReadVariableOp2x
:mnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp:mnist_ae/mnist_enc/separable_conv2d/BiasAdd/ReadVariableOp2
Cmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOpCmnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp2
Emnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_1Emnist_ae/mnist_enc/separable_conv2d/separable_conv2d/ReadVariableOp_12|
<mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp<mnist_ae/mnist_enc/separable_conv2d_1/BiasAdd/ReadVariableOp2
Emnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOpEmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp2
Gmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Gmnist_ae/mnist_enc/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
µ
T
8__inference_global_average_pooling2d_layer_call_fn_95479

inputs
identityÝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_954732
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
¥
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95741

enc_in&
conv2d_95715:
conv2d_95717:0
separable_conv2d_95720:0
separable_conv2d_95722:1$
separable_conv2d_95724:12
separable_conv2d_1_95727:12
separable_conv2d_1_95729:11&
separable_conv2d_1_95731:1(
conv2d_1_95734:11
conv2d_1_95736:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_95715conv2d_95717*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_954972 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_95720separable_conv2d_95722separable_conv2d_95724*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_954252*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95727separable_conv2d_1_95729separable_conv2d_1_95731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_954542,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_95734conv2d_1_95736*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_955282"
 conv2d_1/StatefulPartitionedCall©
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_954732*
(global_average_pooling2d/PartitionedCall
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityê
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
Ô
Ñ
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96533

enc_in)
mnist_enc_96488:
mnist_enc_96490:)
mnist_enc_96492:)
mnist_enc_96494:1
mnist_enc_96496:1)
mnist_enc_96498:1)
mnist_enc_96500:11
mnist_enc_96502:1)
mnist_enc_96504:11
mnist_enc_96506:1)
mnist_dec_96509:)
mnist_dec_96511:1
mnist_dec_96513:1)
mnist_dec_96515:1)
mnist_dec_96517:11
mnist_dec_96519:1)
mnist_dec_96521:1)
mnist_dec_96523:11
mnist_dec_96525:1)
mnist_dec_96527:1
mnist_dec_96529:
identity¢!mnist_dec/StatefulPartitionedCall¢!mnist_enc/StatefulPartitionedCall±
!mnist_enc/StatefulPartitionedCallStatefulPartitionedCallenc_inmnist_enc_96488mnist_enc_96490mnist_enc_96492mnist_enc_96494mnist_enc_96496mnist_enc_96498mnist_enc_96500mnist_enc_96502mnist_enc_96504mnist_enc_96506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_enc_layer_call_and_return_conditional_losses_956352#
!mnist_enc/StatefulPartitionedCall
!mnist_dec/StatefulPartitionedCallStatefulPartitionedCall*mnist_enc/StatefulPartitionedCall:output:0mnist_dec_96509mnist_dec_96511mnist_dec_96513mnist_dec_96515mnist_dec_96517mnist_dec_96519mnist_dec_96521mnist_dec_96523mnist_dec_96525mnist_dec_96527mnist_dec_96529*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_mnist_dec_layer_call_and_return_conditional_losses_960332#
!mnist_dec/StatefulPartitionedCall
IdentityIdentity*mnist_dec/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^mnist_dec/StatefulPartitionedCall"^mnist_enc/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : 2F
!mnist_dec/StatefulPartitionedCall!mnist_dec/StatefulPartitionedCall2F
!mnist_enc/StatefulPartitionedCall!mnist_enc/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
A
enc_in7
serving_default_enc_in:0ÿÿÿÿÿÿÿÿÿE
	mnist_dec8
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ò
Á
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
+ã&call_and_return_all_conditional_losses
ä_default_save_signature
å__call__"
_tf_keras_network
"
_tf_keras_input_layer
ß
layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
trainable_variables
regularization_losses
	variables
	keras_api
+æ&call_and_return_all_conditional_losses
ç__call__"
_tf_keras_network
ù
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
trainable_variables
regularization_losses
	variables
	keras_api
+è&call_and_return_all_conditional_losses
é__call__"
_tf_keras_network

iter

 beta_1

!beta_2
	"decay
#learning_rate$m¹%mº&m»'m¼(m½)m¾*m¿+mÀ,mÁ-mÂ.mÃ/mÄ0mÅ1mÆ2mÇ3mÈ4mÉ5mÊ6mË7mÌ8mÍ$vÎ%vÏ&vÐ'vÑ(vÒ)vÓ*vÔ+vÕ,vÖ-v×.vØ/vÙ0vÚ1vÛ2vÜ3vÝ4vÞ5vß6và7vá8vâ"
tf_deprecated_optimizer
¾
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820"
trackable_list_wrapper
 "
trackable_list_wrapper
¾
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820"
trackable_list_wrapper
Î
9layer_metrics
:metrics
trainable_variables
;layer_regularization_losses

<layers
regularization_losses
	variables
=non_trainable_variables
å__call__
ä_default_save_signature
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
-
êserving_default"
signature_map
½

$kernel
%bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"
_tf_keras_layer
Ý
&depthwise_kernel
'pointwise_kernel
(bias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+í&call_and_return_all_conditional_losses
î__call__"
_tf_keras_layer
Ý
)depthwise_kernel
*pointwise_kernel
+bias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"
_tf_keras_layer
½

,kernel
-bias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"
_tf_keras_layer
§
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"
_tf_keras_layer
f
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9"
trackable_list_wrapper
°
Rlayer_metrics
Smetrics
trainable_variables
Tlayer_regularization_losses

Ulayers
regularization_losses
	variables
Vnon_trainable_variables
ç__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
§
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"
_tf_keras_layer
Ý
.depthwise_kernel
/pointwise_kernel
0bias
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"
_tf_keras_layer
§
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+ù&call_and_return_all_conditional_losses
ú__call__"
_tf_keras_layer
Ý
1depthwise_kernel
2pointwise_kernel
3bias
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
+û&call_and_return_all_conditional_losses
ü__call__"
_tf_keras_layer
§
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"
_tf_keras_layer
Ý
4depthwise_kernel
5pointwise_kernel
6bias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

7kernel
8bias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
n
.0
/1
02
13
24
35
46
57
68
79
810"
trackable_list_wrapper
 "
trackable_list_wrapper
n
.0
/1
02
13
24
35
46
57
68
79
810"
trackable_list_wrapper
°
slayer_metrics
tmetrics
trainable_variables
ulayer_regularization_losses

vlayers
regularization_losses
	variables
wnon_trainable_variables
é__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%2conv2d/kernel
:2conv2d/bias
;:92!separable_conv2d/depthwise_kernel
;:912!separable_conv2d/pointwise_kernel
#:!12separable_conv2d/bias
=:;12#separable_conv2d_1/depthwise_kernel
=:;112#separable_conv2d_1/pointwise_kernel
%:#12separable_conv2d_1/bias
):'112conv2d_1/kernel
:12conv2d_1/bias
=:;2#separable_conv2d_2/depthwise_kernel
=:;12#separable_conv2d_2/pointwise_kernel
%:#12separable_conv2d_2/bias
=:;12#separable_conv2d_3/depthwise_kernel
=:;112#separable_conv2d_3/pointwise_kernel
%:#12separable_conv2d_3/bias
=:;12#separable_conv2d_4/depthwise_kernel
=:;112#separable_conv2d_4/pointwise_kernel
%:#12separable_conv2d_4/bias
):'12conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_dict_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
°
ylayer_metrics
zmetrics
>trainable_variables
{layer_regularization_losses

|layers
?regularization_losses
@	variables
}non_trainable_variables
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
³
~layer_metrics
metrics
Btrainable_variables
 layer_regularization_losses
layers
Cregularization_losses
D	variables
non_trainable_variables
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
µ
layer_metrics
metrics
Ftrainable_variables
 layer_regularization_losses
layers
Gregularization_losses
H	variables
non_trainable_variables
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
µ
layer_metrics
metrics
Jtrainable_variables
 layer_regularization_losses
layers
Kregularization_losses
L	variables
non_trainable_variables
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
Ntrainable_variables
 layer_regularization_losses
layers
Oregularization_losses
P	variables
non_trainable_variables
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
Wtrainable_variables
 layer_regularization_losses
layers
Xregularization_losses
Y	variables
non_trainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
µ
layer_metrics
metrics
[trainable_variables
 layer_regularization_losses
layers
\regularization_losses
]	variables
non_trainable_variables
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
_trainable_variables
 layer_regularization_losses
layers
`regularization_losses
a	variables
 non_trainable_variables
ú__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
5
10
21
32"
trackable_list_wrapper
 "
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
µ
¡layer_metrics
¢metrics
ctrainable_variables
 £layer_regularization_losses
¤layers
dregularization_losses
e	variables
¥non_trainable_variables
ü__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¦layer_metrics
§metrics
gtrainable_variables
 ¨layer_regularization_losses
©layers
hregularization_losses
i	variables
ªnon_trainable_variables
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
µ
«layer_metrics
¬metrics
ktrainable_variables
 ­layer_regularization_losses
®layers
lregularization_losses
m	variables
¯non_trainable_variables
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
µ
°layer_metrics
±metrics
otrainable_variables
 ²layer_regularization_losses
³layers
pregularization_losses
q	variables
´non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
R

µtotal

¶count
·	variables
¸	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
µ0
¶1"
trackable_list_wrapper
.
·	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
@:>2(Adam/separable_conv2d/depthwise_kernel/m
@:>12(Adam/separable_conv2d/pointwise_kernel/m
(:&12Adam/separable_conv2d/bias/m
B:@12*Adam/separable_conv2d_1/depthwise_kernel/m
B:@112*Adam/separable_conv2d_1/pointwise_kernel/m
*:(12Adam/separable_conv2d_1/bias/m
.:,112Adam/conv2d_1/kernel/m
 :12Adam/conv2d_1/bias/m
B:@2*Adam/separable_conv2d_2/depthwise_kernel/m
B:@12*Adam/separable_conv2d_2/pointwise_kernel/m
*:(12Adam/separable_conv2d_2/bias/m
B:@12*Adam/separable_conv2d_3/depthwise_kernel/m
B:@112*Adam/separable_conv2d_3/pointwise_kernel/m
*:(12Adam/separable_conv2d_3/bias/m
B:@12*Adam/separable_conv2d_4/depthwise_kernel/m
B:@112*Adam/separable_conv2d_4/pointwise_kernel/m
*:(12Adam/separable_conv2d_4/bias/m
.:,12Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
@:>2(Adam/separable_conv2d/depthwise_kernel/v
@:>12(Adam/separable_conv2d/pointwise_kernel/v
(:&12Adam/separable_conv2d/bias/v
B:@12*Adam/separable_conv2d_1/depthwise_kernel/v
B:@112*Adam/separable_conv2d_1/pointwise_kernel/v
*:(12Adam/separable_conv2d_1/bias/v
.:,112Adam/conv2d_1/kernel/v
 :12Adam/conv2d_1/bias/v
B:@2*Adam/separable_conv2d_2/depthwise_kernel/v
B:@12*Adam/separable_conv2d_2/pointwise_kernel/v
*:(12Adam/separable_conv2d_2/bias/v
B:@12*Adam/separable_conv2d_3/depthwise_kernel/v
B:@112*Adam/separable_conv2d_3/pointwise_kernel/v
*:(12Adam/separable_conv2d_3/bias/v
B:@12*Adam/separable_conv2d_4/depthwise_kernel/v
B:@112*Adam/separable_conv2d_4/pointwise_kernel/v
*:(12Adam/separable_conv2d_4/bias/v
.:,12Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
Ú2×
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96693
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96798
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96485
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96533À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
å2â
 __inference__wrapped_model_95408½
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *-¢*
(%
enc_inÿÿÿÿÿÿÿÿÿ
î2ë
(__inference_mnist_ae_layer_call_fn_96248
(__inference_mnist_ae_layer_call_fn_96845
(__inference_mnist_ae_layer_call_fn_96892
(__inference_mnist_ae_layer_call_fn_96437À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_mnist_enc_layer_call_and_return_conditional_losses_96936
D__inference_mnist_enc_layer_call_and_return_conditional_losses_96980
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95712
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95741À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
)__inference_mnist_enc_layer_call_fn_95559
)__inference_mnist_enc_layer_call_fn_97005
)__inference_mnist_enc_layer_call_fn_97030
)__inference_mnist_enc_layer_call_fn_95683À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_mnist_dec_layer_call_and_return_conditional_losses_97095
D__inference_mnist_dec_layer_call_and_return_conditional_losses_97160
D__inference_mnist_dec_layer_call_and_return_conditional_losses_96118
D__inference_mnist_dec_layer_call_and_return_conditional_losses_96151À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
)__inference_mnist_dec_layer_call_fn_95955
)__inference_mnist_dec_layer_call_fn_97187
)__inference_mnist_dec_layer_call_fn_97214
)__inference_mnist_dec_layer_call_fn_96085À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÉBÆ
#__inference_signature_wrapper_96588enc_in"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_conv2d_layer_call_and_return_conditional_losses_97225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_conv2d_layer_call_fn_97234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95425×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_separable_conv2d_layer_call_fn_95437×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95454×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
2
2__inference_separable_conv2d_1_layer_call_fn_95466×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
í2ê
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97245¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_1_layer_call_fn_97254¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
»2¸
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_95473à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 2
8__inference_global_average_pooling2d_layer_call_fn_95479à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì2é
B__inference_reshape_layer_call_and_return_conditional_losses_97268¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_reshape_layer_call_fn_97273¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬2©
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_95758×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_separable_conv2d_2_layer_call_fn_95770×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_95783à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_up_sampling2d_layer_call_fn_95789à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_95806×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
2
2__inference_separable_conv2d_3_layer_call_fn_95818×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
²2¯
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_95831à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_up_sampling2d_1_layer_call_fn_95837à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_95854×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
2
2__inference_separable_conv2d_4_layer_call_fn_95866×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
í2ê
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97284¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_2_layer_call_fn_97293¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ´
 __inference__wrapped_model_95408$%&'()*+,-./0123456787¢4
-¢*
(%
enc_inÿÿÿÿÿÿÿÿÿ
ª "=ª:
8
	mnist_dec+(
	mnist_decÿÿÿÿÿÿÿÿÿ³
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97245l,-7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_conv2d_1_layer_call_fn_97254_,-7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1Ø
C__inference_conv2d_2_layer_call_and_return_conditional_losses_9728478I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
(__inference_conv2d_2_layer_call_fn_9729378I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
A__inference_conv2d_layer_call_and_return_conditional_losses_97225l$%7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_conv2d_layer_call_fn_97234_$%7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÜ
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_95473R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
8__inference_global_average_pooling2d_layer_call_fn_95479wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96485$%&'()*+,-./012345678?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 á
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96533$%&'()*+,-./012345678?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96693$%&'()*+,-./012345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ï
C__inference_mnist_ae_layer_call_and_return_conditional_losses_96798$%&'()*+,-./012345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ¹
(__inference_mnist_ae_layer_call_fn_96248$%&'()*+,-./012345678?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
(__inference_mnist_ae_layer_call_fn_96437$%&'()*+,-./012345678?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
(__inference_mnist_ae_layer_call_fn_96845$%&'()*+,-./012345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
(__inference_mnist_ae_layer_call_fn_96892$%&'()*+,-./012345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
D__inference_mnist_dec_layer_call_and_return_conditional_losses_96118./0123456787¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
D__inference_mnist_dec_layer_call_and_return_conditional_losses_96151./0123456787¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ½
D__inference_mnist_dec_layer_call_and_return_conditional_losses_97095u./0123456787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ½
D__inference_mnist_dec_layer_call_and_return_conditional_losses_97160u./0123456787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 §
)__inference_mnist_dec_layer_call_fn_95955z./0123456787¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
)__inference_mnist_dec_layer_call_fn_96085z./0123456787¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
)__inference_mnist_dec_layer_call_fn_97187z./0123456787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
)__inference_mnist_dec_layer_call_fn_97214z./0123456787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95712t
$%&'()*+,-?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 ¼
D__inference_mnist_enc_layer_call_and_return_conditional_losses_95741t
$%&'()*+,-?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 ¼
D__inference_mnist_enc_layer_call_and_return_conditional_losses_96936t
$%&'()*+,-?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 ¼
D__inference_mnist_enc_layer_call_and_return_conditional_losses_96980t
$%&'()*+,-?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_mnist_enc_layer_call_fn_95559g
$%&'()*+,-?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_mnist_enc_layer_call_fn_95683g
$%&'()*+,-?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_mnist_enc_layer_call_fn_97005g
$%&'()*+,-?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_mnist_enc_layer_call_fn_97030g
$%&'()*+,-?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ1¦
B__inference_reshape_layer_call_and_return_conditional_losses_97268`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ~
'__inference_reshape_layer_call_fn_97273S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿã
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95454)*+I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_1_layer_call_fn_95466)*+I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_95758./0I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_2_layer_call_fn_95770./0I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_95806123I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_3_layer_call_fn_95818123I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_95854456I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_4_layer_call_fn_95866456I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1á
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95425&'(I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 ¹
0__inference_separable_conv2d_layer_call_fn_95437&'(I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1Á
#__inference_signature_wrapper_96588$%&'()*+,-./012345678A¢>
¢ 
7ª4
2
enc_in(%
enc_inÿÿÿÿÿÿÿÿÿ"=ª:
8
	mnist_dec+(
	mnist_decÿÿÿÿÿÿÿÿÿí
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_95831R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_up_sampling2d_1_layer_call_fn_95837R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_95783R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_up_sampling2d_layer_call_fn_95789R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
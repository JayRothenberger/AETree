ωΡ
ω
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
Ύ
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
φ
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
 "serve*2.6.02unknown8τι
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
ͺ
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
ͺ
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
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:11*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:1*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:11*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:1*
dtype0
ͺ
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
ͺ
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
ͺ
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
ͺ
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
ͺ
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
ͺ
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
΄
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
΄
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
Έ
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
Έ
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

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:11*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:1*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:11*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:1*
dtype0
Έ
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
Έ
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
Έ
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
Έ
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
Έ
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
Έ
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
΄
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
΄
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
Έ
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
Έ
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

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:11*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:1*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:11*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:1*
dtype0
Έ
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
Έ
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
Έ
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
Έ
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
Έ
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
Έ
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
Σ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueBώ Bφ
Μ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
 
γ
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
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
’
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	variables
trainable_variables
 regularization_losses
!	keras_api
΄
"iter

#beta_1

$beta_2
	%decay
&learning_rate'mΫ(mά)mέ*mή+mί,mΰ-mα.mβ/mγ0mδ1mε2mζ3mη4mθ5mι6mκ7mλ8mμ9mν:mξ;mο<mπ=mρ>mς?mσ'vτ(vυ)vφ*vχ+vψ,vω-vϊ.vϋ/vό0vύ1vώ2v?3v4v5v6v7v8v9v:v;v<v=v>v?v
Ύ
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
Ύ
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
 
­
	variables
@layer_metrics
Alayer_regularization_losses

Blayers
trainable_variables
regularization_losses
Cnon_trainable_variables
Dmetrics
 
h

'kernel
(bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api

)depthwise_kernel
*pointwise_kernel
+bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api

,depthwise_kernel
-pointwise_kernel
.bias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
h

/kernel
0bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
R
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

1kernel
2bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
h

3kernel
4bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
f
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
f
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
 
­
	variables
elayer_metrics
flayer_regularization_losses

glayers
trainable_variables
regularization_losses
hnon_trainable_variables
imetrics
 
R
j	variables
ktrainable_variables
lregularization_losses
m	keras_api

5depthwise_kernel
6pointwise_kernel
7bias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
R
r	variables
strainable_variables
tregularization_losses
u	keras_api

8depthwise_kernel
9pointwise_kernel
:bias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
R
z	variables
{trainable_variables
|regularization_losses
}	keras_api

;depthwise_kernel
<pointwise_kernel
=bias
~	variables
trainable_variables
regularization_losses
	keras_api
l

>kernel
?bias
	variables
trainable_variables
regularization_losses
	keras_api
N
50
61
72
83
94
:5
;6
<7
=8
>9
?10
N
50
61
72
83
94
:5
;6
<7
=8
>9
?10
 
²
	variables
layer_metrics
 layer_regularization_losses
layers
trainable_variables
 regularization_losses
non_trainable_variables
metrics
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
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!separable_conv2d/depthwise_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!separable_conv2d/pointwise_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEseparable_conv2d/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEseparable_conv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEseparable_conv2d_2/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEseparable_conv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEseparable_conv2d_4/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
 

0

'0
(1

'0
(1
 
²
E	variables
layer_metrics
 layer_regularization_losses
layers
Ftrainable_variables
Gregularization_losses
non_trainable_variables
metrics

)0
*1
+2

)0
*1
+2
 
²
I	variables
layer_metrics
 layer_regularization_losses
layers
Jtrainable_variables
Kregularization_losses
non_trainable_variables
metrics

,0
-1
.2

,0
-1
.2
 
²
M	variables
layer_metrics
 layer_regularization_losses
layers
Ntrainable_variables
Oregularization_losses
non_trainable_variables
metrics

/0
01

/0
01
 
²
Q	variables
layer_metrics
 layer_regularization_losses
layers
Rtrainable_variables
Sregularization_losses
non_trainable_variables
metrics
 
 
 
²
U	variables
 layer_metrics
 ‘layer_regularization_losses
’layers
Vtrainable_variables
Wregularization_losses
£non_trainable_variables
€metrics

10
21

10
21
 
²
Y	variables
₯layer_metrics
 ¦layer_regularization_losses
§layers
Ztrainable_variables
[regularization_losses
¨non_trainable_variables
©metrics

30
41

30
41
 
²
]	variables
ͺlayer_metrics
 «layer_regularization_losses
¬layers
^trainable_variables
_regularization_losses
­non_trainable_variables
?metrics
 
 
 
²
a	variables
―layer_metrics
 °layer_regularization_losses
±layers
btrainable_variables
cregularization_losses
²non_trainable_variables
³metrics
 
 
?
0

1
2
3
4
5
6
7
8
 
 
 
 
 
²
j	variables
΄layer_metrics
 ΅layer_regularization_losses
Άlayers
ktrainable_variables
lregularization_losses
·non_trainable_variables
Έmetrics

50
61
72

50
61
72
 
²
n	variables
Ήlayer_metrics
 Ίlayer_regularization_losses
»layers
otrainable_variables
pregularization_losses
Όnon_trainable_variables
½metrics
 
 
 
²
r	variables
Ύlayer_metrics
 Ώlayer_regularization_losses
ΐlayers
strainable_variables
tregularization_losses
Αnon_trainable_variables
Βmetrics

80
91
:2

80
91
:2
 
²
v	variables
Γlayer_metrics
 Δlayer_regularization_losses
Εlayers
wtrainable_variables
xregularization_losses
Ζnon_trainable_variables
Ηmetrics
 
 
 
²
z	variables
Θlayer_metrics
 Ιlayer_regularization_losses
Κlayers
{trainable_variables
|regularization_losses
Λnon_trainable_variables
Μmetrics

;0
<1
=2

;0
<1
=2
 
³
~	variables
Νlayer_metrics
 Ξlayer_regularization_losses
Οlayers
trainable_variables
regularization_losses
Πnon_trainable_variables
Ρmetrics

>0
?1

>0
?1
 
΅
	variables
?layer_metrics
 Σlayer_regularization_losses
Τlayers
trainable_variables
regularization_losses
Υnon_trainable_variables
Φmetrics
 
 
8
0
1
2
3
4
5
6
7
 
 
8

Χtotal

Ψcount
Ω	variables
Ϊ	keras_api
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
Χ0
Ψ1

Ω	variables
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/separable_conv2d/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/separable_conv2d_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/separable_conv2d_2/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/separable_conv2d_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/separable_conv2d_4/bias/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_2/kernel/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_2/bias/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/separable_conv2d/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/separable_conv2d_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/separable_conv2d_2/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/separable_conv2d_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/separable_conv2d_4/bias/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_2/kernel/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_2/bias/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_enc_inPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
λ
StatefulPartitionedCallStatefulPartitionedCallserving_default_enc_inconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_102672
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_2/bias/Read/ReadVariableOp7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOp0Adam/separable_conv2d/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_2/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOp0Adam/separable_conv2d/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_2/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOpConst*_
TinX
V2T	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_103840

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/biastotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/m(Adam/separable_conv2d/depthwise_kernel/m(Adam/separable_conv2d/pointwise_kernel/mAdam/separable_conv2d/bias/m*Adam/separable_conv2d_1/depthwise_kernel/m*Adam/separable_conv2d_1/pointwise_kernel/mAdam/separable_conv2d_1/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m*Adam/separable_conv2d_2/depthwise_kernel/m*Adam/separable_conv2d_2/pointwise_kernel/mAdam/separable_conv2d_2/bias/m*Adam/separable_conv2d_3/depthwise_kernel/m*Adam/separable_conv2d_3/pointwise_kernel/mAdam/separable_conv2d_3/bias/m*Adam/separable_conv2d_4/depthwise_kernel/m*Adam/separable_conv2d_4/pointwise_kernel/mAdam/separable_conv2d_4/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/v(Adam/separable_conv2d/depthwise_kernel/v(Adam/separable_conv2d/pointwise_kernel/vAdam/separable_conv2d/bias/v*Adam/separable_conv2d_1/depthwise_kernel/v*Adam/separable_conv2d_1/pointwise_kernel/vAdam/separable_conv2d_1/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*Adam/separable_conv2d_2/depthwise_kernel/v*Adam/separable_conv2d_2/pointwise_kernel/vAdam/separable_conv2d_2/bias/v*Adam/separable_conv2d_3/depthwise_kernel/v*Adam/separable_conv2d_3/pointwise_kernel/vAdam/separable_conv2d_3/bias/v*Adam/separable_conv2d_4/depthwise_kernel/v*Adam/separable_conv2d_4/pointwise_kernel/vAdam/separable_conv2d_4/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v*^
TinW
U2S*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_104096κ
©	
Λ
1__inference_separable_conv2d_layer_call_fn_101305

inputs!
unknown:#
	unknown_0:1
	unknown_1:1
identity’StatefulPartitionedCall¦
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
GPU2*0J 8 *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_1012932
StatefulPartitionedCall
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
­	
Ν
3__inference_separable_conv2d_4_layer_call_fn_101878

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity’StatefulPartitionedCall¨
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1018662
StatefulPartitionedCall
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
¦
r
)__inference_sampling_layer_call_fn_103516
inputs_0
inputs_1
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
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
GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1014532
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
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????1:?????????122
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????1
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????1
"
_user_specified_name
inputs/1
Υh
Δ
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_103430

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
identity’conv2d_2/BiasAdd/ReadVariableOp’conv2d_2/Conv2D/ReadVariableOp’)separable_conv2d_2/BiasAdd/ReadVariableOp’2separable_conv2d_2/separable_conv2d/ReadVariableOp’4separable_conv2d_2/separable_conv2d/ReadVariableOp_1’)separable_conv2d_3/BiasAdd/ReadVariableOp’2separable_conv2d_3/separable_conv2d/ReadVariableOp’4separable_conv2d_3/separable_conv2d/ReadVariableOp_1’)separable_conv2d_4/BiasAdd/ReadVariableOp’2separable_conv2d_4/separable_conv2d/ReadVariableOp’4separable_conv2d_4/separable_conv2d/ReadVariableOp_1T
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
reshape/Reshape/shape/3κ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapeμ
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpς
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_2/separable_conv2d/dilation_rate―
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativereshape/Reshape:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwise­
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2dΕ
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOpή
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_2/BiasAdd
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
up_sampling2d/Const_1
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_2/Selu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborμ
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpς
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_3/separable_conv2d/dilation_rate?
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise­
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2dΕ
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpή
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_3/BiasAdd
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
:?????????1*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborμ
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpς
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_4/separable_conv2d/dilation_rateΤ
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise­
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2dΕ
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpή
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/BiasAdd
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/Selu°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02 
conv2d_2/Conv2D/ReadVariableOpέ
conv2d_2/Conv2DConv2D%separable_conv2d_4/Selu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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

IdentityΩ
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
ε

)__inference_conv2d_2_layer_call_fn_103560

inputs!
unknown:1
	unknown_0:
identity’StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1019352
StatefulPartitionedCall
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
§

N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_101818

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’separable_conv2d/ReadVariableOp’!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOpΉ
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
separable_conv2d/dilation_rateφ
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingSAME*
strides
2
separable_conv2d/depthwiseσ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity­
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
­	
Ν
3__inference_separable_conv2d_2_layer_call_fn_101782

inputs!
unknown:#
	unknown_0:1
	unknown_1:1
identity’StatefulPartitionedCall¨
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1017702
StatefulPartitionedCall
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
Χ
ύ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_101935

inputs8
conv2d_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Selu
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

s
D__inference_sampling_layer_call_and_return_conditional_losses_103532
inputs_0
inputs_1
identityF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevΝ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2φ^2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????12
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????12
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ym
truedivRealDivinputs_1truediv/y:output:0*
T0*'
_output_shapes
:?????????12	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????12
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????12
mulX
addAddV2mul:z:0inputs_0*
T0*'
_output_shapes
:?????????12
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????1:?????????1:Q M
'
_output_shapes
:?????????1
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????1
"
_user_specified_name
inputs/1
Όe
ΐ
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_103177

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
(conv2d_1_biasadd_readvariableop_resource:16
$dense_matmul_readvariableop_resource:113
%dense_biasadd_readvariableop_resource:18
&dense_1_matmul_readvariableop_resource:115
'dense_1_biasadd_readvariableop_resource:1
identity’conv2d/BiasAdd/ReadVariableOp’conv2d/Conv2D/ReadVariableOp’conv2d_1/BiasAdd/ReadVariableOp’conv2d_1/Conv2D/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’'separable_conv2d/BiasAdd/ReadVariableOp’0separable_conv2d/separable_conv2d/ReadVariableOp’2separable_conv2d/separable_conv2d/ReadVariableOp_1’)separable_conv2d_1/BiasAdd/ReadVariableOp’2separable_conv2d_1/separable_conv2d/ReadVariableOp’4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ͺ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpΈ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D‘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp€
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Seluζ
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpμ
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
/separable_conv2d/separable_conv2d/dilation_rateͺ
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise₯
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2dΏ
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOpΦ
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/BiasAdd
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/Seluμ
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpς
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_1/separable_conv2d/dilation_rateΊ
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise­
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2dΕ
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOpή
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/BiasAdd
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/Selu°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02 
conv2d_1/Conv2D/ReadVariableOpέ
conv2d_1/Conv2DConv2D%separable_conv2d_1/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
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
:?????????12
conv2d_1/BiasAdd{
conv2d_1/SeluSeluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
conv2d_1/Selu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesΟ
global_average_pooling2d/MeanMeanconv2d_1/Selu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12
global_average_pooling2d/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense/MatMul/ReadVariableOp₯
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense/BiasAddj

dense/SeluSeludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12

dense/Selu₯
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense_1/MatMul/ReadVariableOp«
dense_1/MatMulMatMul&global_average_pooling2d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense_1/MatMul€
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02 
dense_1/BiasAdd/ReadVariableOp‘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense_1/BiasAddp
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12
dense_1/Seluj
sampling/ShapeShapedense_1/Selu:activations:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sampling/random_normal/stddevι
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2ώ2-
+sampling/random_normal/RandomStandardNormalΟ
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????12
sampling/random_normal/mul±
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????12
sampling/random_normalm
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sampling/truediv/y
sampling/truedivRealDivdense_1/Selu:activations:0sampling/truediv/y:output:0*
T0*'
_output_shapes
:?????????12
sampling/truedivk
sampling/ExpExpsampling/truediv:z:0*
T0*'
_output_shapes
:?????????12
sampling/Exp
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*'
_output_shapes
:?????????12
sampling/mul
sampling/addAddV2sampling/mul:z:0dense/Selu:activations:0*
T0*'
_output_shapes
:?????????12
sampling/addk
IdentityIdentitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityψ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
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


-__inference_mnist_ae_var_layer_call_fn_102497

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1$

unknown_13:$

unknown_14:1

unknown_15:1$

unknown_16:1$

unknown_17:11

unknown_18:1$

unknown_19:1$

unknown_20:11

unknown_21:1$

unknown_22:1

unknown_23:
identity’StatefulPartitionedCallΞ
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
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_1023892
StatefulPartitionedCall
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
Κ-

I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101713

enc_in'
conv2d_101676:
conv2d_101678:1
separable_conv2d_101681:1
separable_conv2d_101683:1%
separable_conv2d_101685:13
separable_conv2d_1_101688:13
separable_conv2d_1_101690:11'
separable_conv2d_1_101692:1)
conv2d_1_101695:11
conv2d_1_101697:1
dense_101701:11
dense_101703:1 
dense_1_101706:11
dense_1_101708:1
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’(separable_conv2d/StatefulPartitionedCall’*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_101676conv2d_101678*
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
GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1013652 
conv2d/StatefulPartitionedCall
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_101681separable_conv2d_101683separable_conv2d_101685*
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
GPU2*0J 8 *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_1012932*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_101688separable_conv2d_1_101690separable_conv2d_1_101692*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1013222,
*separable_conv2d_1/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_101695conv2d_1_101697*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1013962"
 conv2d_1/StatefulPartitionedCallͺ
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1013412*
(global_average_pooling2d/PartitionedCall³
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_101701dense_101703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1014142
dense/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_101706dense_1_101708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1014312!
dense_1/StatefulPartitionedCallΊ
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1014532"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

IdentityΟ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
»¨
Ί!
!__inference__wrapped_model_101276

enc_inZ
@mnist_ae_var_mnist_enc_var_conv2d_conv2d_readvariableop_resource:O
Amnist_ae_var_mnist_enc_var_conv2d_biasadd_readvariableop_resource:n
Tmnist_ae_var_mnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_resource:p
Vmnist_ae_var_mnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_1_resource:1Y
Kmnist_ae_var_mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource:1p
Vmnist_ae_var_mnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource:1r
Xmnist_ae_var_mnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11[
Mmnist_ae_var_mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource:1\
Bmnist_ae_var_mnist_enc_var_conv2d_1_conv2d_readvariableop_resource:11Q
Cmnist_ae_var_mnist_enc_var_conv2d_1_biasadd_readvariableop_resource:1Q
?mnist_ae_var_mnist_enc_var_dense_matmul_readvariableop_resource:11N
@mnist_ae_var_mnist_enc_var_dense_biasadd_readvariableop_resource:1S
Amnist_ae_var_mnist_enc_var_dense_1_matmul_readvariableop_resource:11P
Bmnist_ae_var_mnist_enc_var_dense_1_biasadd_readvariableop_resource:1p
Vmnist_ae_var_mnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource:r
Xmnist_ae_var_mnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:1[
Mmnist_ae_var_mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource:1p
Vmnist_ae_var_mnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource:1r
Xmnist_ae_var_mnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource:11[
Mmnist_ae_var_mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource:1p
Vmnist_ae_var_mnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource:1r
Xmnist_ae_var_mnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource:11[
Mmnist_ae_var_mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource:1\
Bmnist_ae_var_mnist_dec_var_conv2d_2_conv2d_readvariableop_resource:1Q
Cmnist_ae_var_mnist_dec_var_conv2d_2_biasadd_readvariableop_resource:
identity’:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp’9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp’Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp’Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp’Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1’Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp’Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp’Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1’Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp’Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp’Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1’8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp’7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp’:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp’9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp’7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp’6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp’9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp’8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp’Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp’Kmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp’Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1’Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp’Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp’Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ϋ
7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOpReadVariableOp@mnist_ae_var_mnist_enc_var_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp
(mnist_ae_var/mnist_enc_var/conv2d/Conv2DConv2Denc_in?mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2*
(mnist_ae_var/mnist_enc_var/conv2d/Conv2Dς
8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOpReadVariableOpAmnist_ae_var_mnist_enc_var_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp
)mnist_ae_var/mnist_enc_var/conv2d/BiasAddBiasAdd1mnist_ae_var/mnist_enc_var/conv2d/Conv2D:output:0@mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2+
)mnist_ae_var/mnist_enc_var/conv2d/BiasAddΖ
&mnist_ae_var/mnist_enc_var/conv2d/SeluSelu2mnist_ae_var/mnist_enc_var/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2(
&mnist_ae_var/mnist_enc_var/conv2d/Selu·
Kmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpTmnist_ae_var_mnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02M
Kmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp½
Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpVmnist_ae_var_mnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1α
Bmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2D
Bmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/Shapeι
Jmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2L
Jmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rate
Fmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative4mnist_ae_var/mnist_enc_var/conv2d/Selu:activations:0Smnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2H
Fmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/depthwise
<mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2dConv2DOmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/depthwise:output:0Umnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2>
<mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d
Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOpKmnist_ae_var_mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02D
Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpΒ
3mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAddBiasAddEmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d:output:0Jmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????125
3mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAddδ
0mnist_ae_var/mnist_enc_var/separable_conv2d/SeluSelu<mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????122
0mnist_ae_var/mnist_enc_var/separable_conv2d/Selu½
Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpΓ
Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02Q
Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ε
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2F
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/Shapeν
Lmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rate¦
Hmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative>mnist_ae_var/mnist_enc_var/separable_conv2d/Selu:activations:0Umnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise
>mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2dConv2DQmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpΚ
5mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAddBiasAddGmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d:output:0Lmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????127
5mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAddκ
2mnist_ae_var/mnist_enc_var/separable_conv2d_1/SeluSelu>mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????124
2mnist_ae_var/mnist_enc_var/separable_conv2d_1/Selu
9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpReadVariableOpBmnist_ae_var_mnist_enc_var_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02;
9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpΙ
*mnist_ae_var/mnist_enc_var/conv2d_1/Conv2DConv2D@mnist_ae_var/mnist_enc_var/separable_conv2d_1/Selu:activations:0Amnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2,
*mnist_ae_var/mnist_enc_var/conv2d_1/Conv2Dψ
:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpCmnist_ae_var_mnist_enc_var_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02<
:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp
+mnist_ae_var/mnist_enc_var/conv2d_1/BiasAddBiasAdd3mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D:output:0Bmnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12-
+mnist_ae_var/mnist_enc_var/conv2d_1/BiasAddΜ
(mnist_ae_var/mnist_enc_var/conv2d_1/SeluSelu4mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12*
(mnist_ae_var/mnist_enc_var/conv2d_1/Seluι
Jmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2L
Jmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean/reduction_indices»
8mnist_ae_var/mnist_enc_var/global_average_pooling2d/MeanMean6mnist_ae_var/mnist_enc_var/conv2d_1/Selu:activations:0Smnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12:
8mnist_ae_var/mnist_enc_var/global_average_pooling2d/Meanπ
6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOpReadVariableOp?mnist_ae_var_mnist_enc_var_dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype028
6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp
'mnist_ae_var/mnist_enc_var/dense/MatMulMatMulAmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean:output:0>mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12)
'mnist_ae_var/mnist_enc_var/dense/MatMulο
7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOpReadVariableOp@mnist_ae_var_mnist_enc_var_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp
(mnist_ae_var/mnist_enc_var/dense/BiasAddBiasAdd1mnist_ae_var/mnist_enc_var/dense/MatMul:product:0?mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12*
(mnist_ae_var/mnist_enc_var/dense/BiasAdd»
%mnist_ae_var/mnist_enc_var/dense/SeluSelu1mnist_ae_var/mnist_enc_var/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12'
%mnist_ae_var/mnist_enc_var/dense/Seluφ
8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOpReadVariableOpAmnist_ae_var_mnist_enc_var_dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02:
8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp
)mnist_ae_var/mnist_enc_var/dense_1/MatMulMatMulAmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean:output:0@mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12+
)mnist_ae_var/mnist_enc_var/dense_1/MatMulυ
9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOpReadVariableOpBmnist_ae_var_mnist_enc_var_dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02;
9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp
*mnist_ae_var/mnist_enc_var/dense_1/BiasAddBiasAdd3mnist_ae_var/mnist_enc_var/dense_1/MatMul:product:0Amnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12,
*mnist_ae_var/mnist_enc_var/dense_1/BiasAddΑ
'mnist_ae_var/mnist_enc_var/dense_1/SeluSelu3mnist_ae_var/mnist_enc_var/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12)
'mnist_ae_var/mnist_enc_var/dense_1/Selu»
)mnist_ae_var/mnist_enc_var/sampling/ShapeShape5mnist_ae_var/mnist_enc_var/dense_1/Selu:activations:0*
T0*
_output_shapes
:2+
)mnist_ae_var/mnist_enc_var/sampling/Shape΅
6mnist_ae_var/mnist_enc_var/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6mnist_ae_var/mnist_enc_var/sampling/random_normal/meanΉ
8mnist_ae_var/mnist_enc_var/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8mnist_ae_var/mnist_enc_var/sampling/random_normal/stddevΉ
Fmnist_ae_var/mnist_enc_var/sampling/random_normal/RandomStandardNormalRandomStandardNormal2mnist_ae_var/mnist_enc_var/sampling/Shape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2χΨE2H
Fmnist_ae_var/mnist_enc_var/sampling/random_normal/RandomStandardNormal»
5mnist_ae_var/mnist_enc_var/sampling/random_normal/mulMulOmnist_ae_var/mnist_enc_var/sampling/random_normal/RandomStandardNormal:output:0Amnist_ae_var/mnist_enc_var/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????127
5mnist_ae_var/mnist_enc_var/sampling/random_normal/mul
1mnist_ae_var/mnist_enc_var/sampling/random_normalAddV29mnist_ae_var/mnist_enc_var/sampling/random_normal/mul:z:0?mnist_ae_var/mnist_enc_var/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????123
1mnist_ae_var/mnist_enc_var/sampling/random_normal£
-mnist_ae_var/mnist_enc_var/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-mnist_ae_var/mnist_enc_var/sampling/truediv/y
+mnist_ae_var/mnist_enc_var/sampling/truedivRealDiv5mnist_ae_var/mnist_enc_var/dense_1/Selu:activations:06mnist_ae_var/mnist_enc_var/sampling/truediv/y:output:0*
T0*'
_output_shapes
:?????????12-
+mnist_ae_var/mnist_enc_var/sampling/truedivΌ
'mnist_ae_var/mnist_enc_var/sampling/ExpExp/mnist_ae_var/mnist_enc_var/sampling/truediv:z:0*
T0*'
_output_shapes
:?????????12)
'mnist_ae_var/mnist_enc_var/sampling/Expο
'mnist_ae_var/mnist_enc_var/sampling/mulMul5mnist_ae_var/mnist_enc_var/sampling/random_normal:z:0+mnist_ae_var/mnist_enc_var/sampling/Exp:y:0*
T0*'
_output_shapes
:?????????12)
'mnist_ae_var/mnist_enc_var/sampling/mulο
'mnist_ae_var/mnist_enc_var/sampling/addAddV2+mnist_ae_var/mnist_enc_var/sampling/mul:z:03mnist_ae_var/mnist_enc_var/dense/Selu:activations:0*
T0*'
_output_shapes
:?????????12)
'mnist_ae_var/mnist_enc_var/sampling/add―
(mnist_ae_var/mnist_dec_var/reshape/ShapeShape+mnist_ae_var/mnist_enc_var/sampling/add:z:0*
T0*
_output_shapes
:2*
(mnist_ae_var/mnist_dec_var/reshape/ShapeΊ
6mnist_ae_var/mnist_dec_var/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6mnist_ae_var/mnist_dec_var/reshape/strided_slice/stackΎ
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_1Ύ
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_2΄
0mnist_ae_var/mnist_dec_var/reshape/strided_sliceStridedSlice1mnist_ae_var/mnist_dec_var/reshape/Shape:output:0?mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack:output:0Amnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_1:output:0Amnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0mnist_ae_var/mnist_dec_var/reshape/strided_sliceͺ
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/1ͺ
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/2ͺ
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/3
0mnist_ae_var/mnist_dec_var/reshape/Reshape/shapePack9mnist_ae_var/mnist_dec_var/reshape/strided_slice:output:0;mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/1:output:0;mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/2:output:0;mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0mnist_ae_var/mnist_dec_var/reshape/Reshape/shape
*mnist_ae_var/mnist_dec_var/reshape/ReshapeReshape+mnist_ae_var/mnist_enc_var/sampling/add:z:09mnist_ae_var/mnist_dec_var/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2,
*mnist_ae_var/mnist_dec_var/reshape/Reshape½
Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02O
Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpΓ
Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02Q
Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ε
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/Shapeν
Lmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rate
Hmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative3mnist_ae_var/mnist_dec_var/reshape/Reshape:output:0Umnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise
>mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DQmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpΚ
5mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAddBiasAddGmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0Lmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????127
5mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAddκ
2mnist_ae_var/mnist_dec_var/separable_conv2d_2/SeluSelu>mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????124
2mnist_ae_var/mnist_dec_var/separable_conv2d_2/Selu±
.mnist_ae_var/mnist_dec_var/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      20
.mnist_ae_var/mnist_dec_var/up_sampling2d/Const΅
0mnist_ae_var/mnist_dec_var/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      22
0mnist_ae_var/mnist_dec_var/up_sampling2d/Const_1ό
,mnist_ae_var/mnist_dec_var/up_sampling2d/mulMul7mnist_ae_var/mnist_dec_var/up_sampling2d/Const:output:09mnist_ae_var/mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2.
,mnist_ae_var/mnist_dec_var/up_sampling2d/mulο
Emnist_ae_var/mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor@mnist_ae_var/mnist_dec_var/separable_conv2d_2/Selu:activations:00mnist_ae_var/mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2G
Emnist_ae_var/mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor½
Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpΓ
Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02Q
Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ε
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/Shapeν
Lmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateΎ
Hmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeVmnist_ae_var/mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Umnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise
>mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DQmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpΚ
5mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAddBiasAddGmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0Lmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????127
5mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAddκ
2mnist_ae_var/mnist_dec_var/separable_conv2d_3/SeluSelu>mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????124
2mnist_ae_var/mnist_dec_var/separable_conv2d_3/Selu΅
0mnist_ae_var/mnist_dec_var/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      22
0mnist_ae_var/mnist_dec_var/up_sampling2d_1/ConstΉ
2mnist_ae_var/mnist_dec_var/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      24
2mnist_ae_var/mnist_dec_var/up_sampling2d_1/Const_1
.mnist_ae_var/mnist_dec_var/up_sampling2d_1/mulMul9mnist_ae_var/mnist_dec_var/up_sampling2d_1/Const:output:0;mnist_ae_var/mnist_dec_var/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:20
.mnist_ae_var/mnist_dec_var/up_sampling2d_1/mulυ
Gmnist_ae_var/mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor@mnist_ae_var/mnist_dec_var/separable_conv2d_3/Selu:activations:02mnist_ae_var/mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2I
Gmnist_ae_var/mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor½
Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpΓ
Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02Q
Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ε
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/Shapeν
Lmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateΐ
Hmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeXmnist_ae_var/mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Umnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise
>mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DQmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpΚ
5mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAddBiasAddGmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0Lmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????127
5mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAddκ
2mnist_ae_var/mnist_dec_var/separable_conv2d_4/SeluSelu>mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????124
2mnist_ae_var/mnist_dec_var/separable_conv2d_4/Selu
9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOpBmnist_ae_var_mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02;
9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpΙ
*mnist_ae_var/mnist_dec_var/conv2d_2/Conv2DConv2D@mnist_ae_var/mnist_dec_var/separable_conv2d_4/Selu:activations:0Amnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2,
*mnist_ae_var/mnist_dec_var/conv2d_2/Conv2Dψ
:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpCmnist_ae_var_mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp
+mnist_ae_var/mnist_dec_var/conv2d_2/BiasAddBiasAdd3mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D:output:0Bmnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2-
+mnist_ae_var/mnist_dec_var/conv2d_2/BiasAddΜ
(mnist_ae_var/mnist_dec_var/conv2d_2/SeluSelu4mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2*
(mnist_ae_var/mnist_dec_var/conv2d_2/Selu
IdentityIdentity6mnist_ae_var/mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity¦
NoOpNoOp;^mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:^mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpE^mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1E^mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1E^mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_19^mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp8^mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp;^mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:^mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp8^mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp7^mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp:^mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp9^mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOpC^mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpL^mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpN^mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1E^mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2x
:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp2v
9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp2
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp2’
Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_12
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp2’
Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_12
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp2’
Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_12t
8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp2r
7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp2x
:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp2v
9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp2r
7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp2p
6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp2v
9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp2t
8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp2
Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpBmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp2
Kmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpKmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp2
Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_12
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp2’
Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
λ
Μ
.__inference_mnist_dec_var_layer_call_fn_102097

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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1020452
StatefulPartitionedCall
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
ν
ί	
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102609

enc_in.
mnist_enc_var_102556:"
mnist_enc_var_102558:.
mnist_enc_var_102560:.
mnist_enc_var_102562:1"
mnist_enc_var_102564:1.
mnist_enc_var_102566:1.
mnist_enc_var_102568:11"
mnist_enc_var_102570:1.
mnist_enc_var_102572:11"
mnist_enc_var_102574:1&
mnist_enc_var_102576:11"
mnist_enc_var_102578:1&
mnist_enc_var_102580:11"
mnist_enc_var_102582:1.
mnist_dec_var_102585:.
mnist_dec_var_102587:1"
mnist_dec_var_102589:1.
mnist_dec_var_102591:1.
mnist_dec_var_102593:11"
mnist_dec_var_102595:1.
mnist_dec_var_102597:1.
mnist_dec_var_102599:11"
mnist_dec_var_102601:1.
mnist_dec_var_102603:1"
mnist_dec_var_102605:
identity’%mnist_dec_var/StatefulPartitionedCall’%mnist_enc_var/StatefulPartitionedCallΠ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallenc_inmnist_enc_var_102556mnist_enc_var_102558mnist_enc_var_102560mnist_enc_var_102562mnist_enc_var_102564mnist_enc_var_102566mnist_enc_var_102568mnist_enc_var_102570mnist_enc_var_102572mnist_enc_var_102574mnist_enc_var_102576mnist_enc_var_102578mnist_enc_var_102580mnist_enc_var_102582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1016092'
%mnist_enc_var/StatefulPartitionedCallΚ
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_102585mnist_dec_var_102587mnist_dec_var_102589mnist_dec_var_102591mnist_dec_var_102593mnist_dec_var_102595mnist_dec_var_102597mnist_dec_var_102599mnist_dec_var_102601mnist_dec_var_102603mnist_dec_var_102605*
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1020452'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?

ς
A__inference_dense_layer_call_and_return_conditional_losses_103490

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
ν
ί	
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102389

inputs.
mnist_enc_var_102336:"
mnist_enc_var_102338:.
mnist_enc_var_102340:.
mnist_enc_var_102342:1"
mnist_enc_var_102344:1.
mnist_enc_var_102346:1.
mnist_enc_var_102348:11"
mnist_enc_var_102350:1.
mnist_enc_var_102352:11"
mnist_enc_var_102354:1&
mnist_enc_var_102356:11"
mnist_enc_var_102358:1&
mnist_enc_var_102360:11"
mnist_enc_var_102362:1.
mnist_dec_var_102365:.
mnist_dec_var_102367:1"
mnist_dec_var_102369:1.
mnist_dec_var_102371:1.
mnist_dec_var_102373:11"
mnist_dec_var_102375:1.
mnist_dec_var_102377:1.
mnist_dec_var_102379:11"
mnist_dec_var_102381:1.
mnist_dec_var_102383:1"
mnist_dec_var_102385:
identity’%mnist_dec_var/StatefulPartitionedCall’%mnist_enc_var/StatefulPartitionedCallΠ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallinputsmnist_enc_var_102336mnist_enc_var_102338mnist_enc_var_102340mnist_enc_var_102342mnist_enc_var_102344mnist_enc_var_102346mnist_enc_var_102348mnist_enc_var_102350mnist_enc_var_102352mnist_enc_var_102354mnist_enc_var_102356mnist_enc_var_102358mnist_enc_var_102360mnist_enc_var_102362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1016092'
%mnist_enc_var/StatefulPartitionedCallΚ
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_102365mnist_dec_var_102367mnist_dec_var_102369mnist_dec_var_102371mnist_dec_var_102373mnist_dec_var_102375mnist_dec_var_102377mnist_dec_var_102379mnist_dec_var_102381mnist_dec_var_102383mnist_dec_var_102385*
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1020452'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
κ
ύ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_103470

inputs8
conv2d_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
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
Κ-

I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101753

enc_in'
conv2d_101716:
conv2d_101718:1
separable_conv2d_101721:1
separable_conv2d_101723:1%
separable_conv2d_101725:13
separable_conv2d_1_101728:13
separable_conv2d_1_101730:11'
separable_conv2d_1_101732:1)
conv2d_1_101735:11
conv2d_1_101737:1
dense_101741:11
dense_101743:1 
dense_1_101746:11
dense_1_101748:1
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’(separable_conv2d/StatefulPartitionedCall’*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_101716conv2d_101718*
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
GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1013652 
conv2d/StatefulPartitionedCall
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_101721separable_conv2d_101723separable_conv2d_101725*
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
GPU2*0J 8 *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_1012932*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_101728separable_conv2d_1_101730separable_conv2d_1_101732*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1013222,
*separable_conv2d_1/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_101735conv2d_1_101737*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1013962"
 conv2d_1/StatefulPartitionedCallͺ
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1013412*
(global_average_pooling2d/PartitionedCall³
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_101741dense_101743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1014142
dense/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_101746dense_1_101748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1014312!
dense_1/StatefulPartitionedCallΊ
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1014532"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

IdentityΟ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in

τ
C__inference_dense_1_layer_call_and_return_conditional_losses_103510

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
­	
Ν
3__inference_separable_conv2d_3_layer_call_fn_101830

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity’StatefulPartitionedCall¨
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1018182
StatefulPartitionedCall
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
­	
Ν
3__inference_separable_conv2d_1_layer_call_fn_101334

inputs!
unknown:1#
	unknown_0:11
	unknown_1:1
identity’StatefulPartitionedCall¨
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1013222
StatefulPartitionedCall
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
Χ
ύ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_103571

inputs8
conv2d_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Selu
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
Ω'

I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_102045

inputs3
separable_conv2d_2_102016:3
separable_conv2d_2_102018:1'
separable_conv2d_2_102020:13
separable_conv2d_3_102024:13
separable_conv2d_3_102026:11'
separable_conv2d_3_102028:13
separable_conv2d_4_102032:13
separable_conv2d_4_102034:11'
separable_conv2d_4_102036:1)
conv2d_2_102039:1
conv2d_2_102041:
identity’ conv2d_2/StatefulPartitionedCall’*separable_conv2d_2/StatefulPartitionedCall’*separable_conv2d_3/StatefulPartitionedCall’*separable_conv2d_4/StatefulPartitionedCallά
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
GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1018992
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_102016separable_conv2d_2_102018separable_conv2d_2_102020*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1017702,
*separable_conv2d_2/StatefulPartitionedCall­
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
GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1017952
up_sampling2d/PartitionedCall 
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_102024separable_conv2d_3_102026separable_conv2d_3_102028*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1018182,
*separable_conv2d_3/StatefulPartitionedCall³
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
GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1018432!
up_sampling2d_1/PartitionedCall’
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_102032separable_conv2d_4_102034separable_conv2d_4_102036*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1018662,
*separable_conv2d_4/StatefulPartitionedCallή
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_102039conv2d_2_102041*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1019352"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityψ
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
θ
ϋ
B__inference_conv2d_layer_call_and_return_conditional_losses_103450

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
θ
ϋ
B__inference_conv2d_layer_call_and_return_conditional_losses_101365

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
Ϊ
L
0__inference_up_sampling2d_1_layer_call_fn_101849

inputs
identityο
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
GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1018432
PartitionedCall
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
ΐ

.__inference_mnist_enc_var_layer_call_fn_101487

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1014562
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
Κ-

I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101609

inputs'
conv2d_101572:
conv2d_101574:1
separable_conv2d_101577:1
separable_conv2d_101579:1%
separable_conv2d_101581:13
separable_conv2d_1_101584:13
separable_conv2d_1_101586:11'
separable_conv2d_1_101588:1)
conv2d_1_101591:11
conv2d_1_101593:1
dense_101597:11
dense_101599:1 
dense_1_101602:11
dense_1_101604:1
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’(separable_conv2d/StatefulPartitionedCall’*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_101572conv2d_101574*
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
GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1013652 
conv2d/StatefulPartitionedCall
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_101577separable_conv2d_101579separable_conv2d_101581*
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
GPU2*0J 8 *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_1012932*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_101584separable_conv2d_1_101586separable_conv2d_1_101588*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1013222,
*separable_conv2d_1/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_101591conv2d_1_101593*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1013962"
 conv2d_1/StatefulPartitionedCallͺ
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1013412*
(global_average_pooling2d/PartitionedCall³
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_101597dense_101599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1014142
dense/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_101602dense_1_101604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1014312!
dense_1/StatefulPartitionedCallΊ
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1014532"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

IdentityΟ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Α
ώ
$__inference_signature_wrapper_102672

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1$

unknown_13:$

unknown_14:1

unknown_15:1$

unknown_16:1$

unknown_17:11

unknown_18:1$

unknown_19:1$

unknown_20:11

unknown_21:1$

unknown_22:1

unknown_23:
identity’StatefulPartitionedCall
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
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_1012762
StatefulPartitionedCall
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
ξ

&__inference_dense_layer_call_fn_103479

inputs
unknown:11
	unknown_0:1
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1014142
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
ν
ί	
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102553

enc_in.
mnist_enc_var_102500:"
mnist_enc_var_102502:.
mnist_enc_var_102504:.
mnist_enc_var_102506:1"
mnist_enc_var_102508:1.
mnist_enc_var_102510:1.
mnist_enc_var_102512:11"
mnist_enc_var_102514:1.
mnist_enc_var_102516:11"
mnist_enc_var_102518:1&
mnist_enc_var_102520:11"
mnist_enc_var_102522:1&
mnist_enc_var_102524:11"
mnist_enc_var_102526:1.
mnist_dec_var_102529:.
mnist_dec_var_102531:1"
mnist_dec_var_102533:1.
mnist_dec_var_102535:1.
mnist_dec_var_102537:11"
mnist_dec_var_102539:1.
mnist_dec_var_102541:1.
mnist_dec_var_102543:11"
mnist_dec_var_102545:1.
mnist_dec_var_102547:1"
mnist_dec_var_102549:
identity’%mnist_dec_var/StatefulPartitionedCall’%mnist_enc_var/StatefulPartitionedCallΠ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallenc_inmnist_enc_var_102500mnist_enc_var_102502mnist_enc_var_102504mnist_enc_var_102506mnist_enc_var_102508mnist_enc_var_102510mnist_enc_var_102512mnist_enc_var_102514mnist_enc_var_102516mnist_enc_var_102518mnist_enc_var_102520mnist_enc_var_102522mnist_enc_var_102524mnist_enc_var_102526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1014562'
%mnist_enc_var/StatefulPartitionedCallΚ
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_102529mnist_dec_var_102531mnist_dec_var_102533mnist_dec_var_102535mnist_dec_var_102537mnist_dec_var_102539mnist_dec_var_102541mnist_dec_var_102543mnist_dec_var_102545mnist_dec_var_102547mnist_dec_var_102549*
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1019422'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
§

N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_101322

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’separable_conv2d/ReadVariableOp’!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOpΉ
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
separable_conv2d/dilation_rateφ
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingSAME*
strides
2
separable_conv2d/depthwiseσ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity­
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


'__inference_conv2d_layer_call_fn_103439

inputs!
unknown:
	unknown_0:
identity’StatefulPartitionedCallύ
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
GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1013652
StatefulPartitionedCall
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
Ω'

I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_102163

dec_in3
separable_conv2d_2_102134:3
separable_conv2d_2_102136:1'
separable_conv2d_2_102138:13
separable_conv2d_3_102142:13
separable_conv2d_3_102144:11'
separable_conv2d_3_102146:13
separable_conv2d_4_102150:13
separable_conv2d_4_102152:11'
separable_conv2d_4_102154:1)
conv2d_2_102157:1
conv2d_2_102159:
identity’ conv2d_2/StatefulPartitionedCall’*separable_conv2d_2/StatefulPartitionedCall’*separable_conv2d_3/StatefulPartitionedCall’*separable_conv2d_4/StatefulPartitionedCallά
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
GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1018992
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_102134separable_conv2d_2_102136separable_conv2d_2_102138*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1017702,
*separable_conv2d_2/StatefulPartitionedCall­
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
GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1017952
up_sampling2d/PartitionedCall 
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_102142separable_conv2d_3_102144separable_conv2d_3_102146*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1018182,
*separable_conv2d_3/StatefulPartitionedCall³
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
GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1018432!
up_sampling2d_1/PartitionedCall’
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_102150separable_conv2d_4_102152separable_conv2d_4_102154*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1018662,
*separable_conv2d_4/StatefulPartitionedCallή
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_102157conv2d_2_102159*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1019352"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityψ
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
γ
:
"__inference__traced_restore_104096
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
!assignvariableop_14_conv2d_1_bias:12
 assignvariableop_15_dense_kernel:11,
assignvariableop_16_dense_bias:14
"assignvariableop_17_dense_1_kernel:11.
 assignvariableop_18_dense_1_bias:1Q
7assignvariableop_19_separable_conv2d_2_depthwise_kernel:Q
7assignvariableop_20_separable_conv2d_2_pointwise_kernel:19
+assignvariableop_21_separable_conv2d_2_bias:1Q
7assignvariableop_22_separable_conv2d_3_depthwise_kernel:1Q
7assignvariableop_23_separable_conv2d_3_pointwise_kernel:119
+assignvariableop_24_separable_conv2d_3_bias:1Q
7assignvariableop_25_separable_conv2d_4_depthwise_kernel:1Q
7assignvariableop_26_separable_conv2d_4_pointwise_kernel:119
+assignvariableop_27_separable_conv2d_4_bias:1=
#assignvariableop_28_conv2d_2_kernel:1/
!assignvariableop_29_conv2d_2_bias:#
assignvariableop_30_total: #
assignvariableop_31_count: B
(assignvariableop_32_adam_conv2d_kernel_m:4
&assignvariableop_33_adam_conv2d_bias_m:V
<assignvariableop_34_adam_separable_conv2d_depthwise_kernel_m:V
<assignvariableop_35_adam_separable_conv2d_pointwise_kernel_m:1>
0assignvariableop_36_adam_separable_conv2d_bias_m:1X
>assignvariableop_37_adam_separable_conv2d_1_depthwise_kernel_m:1X
>assignvariableop_38_adam_separable_conv2d_1_pointwise_kernel_m:11@
2assignvariableop_39_adam_separable_conv2d_1_bias_m:1D
*assignvariableop_40_adam_conv2d_1_kernel_m:116
(assignvariableop_41_adam_conv2d_1_bias_m:19
'assignvariableop_42_adam_dense_kernel_m:113
%assignvariableop_43_adam_dense_bias_m:1;
)assignvariableop_44_adam_dense_1_kernel_m:115
'assignvariableop_45_adam_dense_1_bias_m:1X
>assignvariableop_46_adam_separable_conv2d_2_depthwise_kernel_m:X
>assignvariableop_47_adam_separable_conv2d_2_pointwise_kernel_m:1@
2assignvariableop_48_adam_separable_conv2d_2_bias_m:1X
>assignvariableop_49_adam_separable_conv2d_3_depthwise_kernel_m:1X
>assignvariableop_50_adam_separable_conv2d_3_pointwise_kernel_m:11@
2assignvariableop_51_adam_separable_conv2d_3_bias_m:1X
>assignvariableop_52_adam_separable_conv2d_4_depthwise_kernel_m:1X
>assignvariableop_53_adam_separable_conv2d_4_pointwise_kernel_m:11@
2assignvariableop_54_adam_separable_conv2d_4_bias_m:1D
*assignvariableop_55_adam_conv2d_2_kernel_m:16
(assignvariableop_56_adam_conv2d_2_bias_m:B
(assignvariableop_57_adam_conv2d_kernel_v:4
&assignvariableop_58_adam_conv2d_bias_v:V
<assignvariableop_59_adam_separable_conv2d_depthwise_kernel_v:V
<assignvariableop_60_adam_separable_conv2d_pointwise_kernel_v:1>
0assignvariableop_61_adam_separable_conv2d_bias_v:1X
>assignvariableop_62_adam_separable_conv2d_1_depthwise_kernel_v:1X
>assignvariableop_63_adam_separable_conv2d_1_pointwise_kernel_v:11@
2assignvariableop_64_adam_separable_conv2d_1_bias_v:1D
*assignvariableop_65_adam_conv2d_1_kernel_v:116
(assignvariableop_66_adam_conv2d_1_bias_v:19
'assignvariableop_67_adam_dense_kernel_v:113
%assignvariableop_68_adam_dense_bias_v:1;
)assignvariableop_69_adam_dense_1_kernel_v:115
'assignvariableop_70_adam_dense_1_bias_v:1X
>assignvariableop_71_adam_separable_conv2d_2_depthwise_kernel_v:X
>assignvariableop_72_adam_separable_conv2d_2_pointwise_kernel_v:1@
2assignvariableop_73_adam_separable_conv2d_2_bias_v:1X
>assignvariableop_74_adam_separable_conv2d_3_depthwise_kernel_v:1X
>assignvariableop_75_adam_separable_conv2d_3_pointwise_kernel_v:11@
2assignvariableop_76_adam_separable_conv2d_3_bias_v:1X
>assignvariableop_77_adam_separable_conv2d_4_depthwise_kernel_v:1X
>assignvariableop_78_adam_separable_conv2d_4_pointwise_kernel_v:11@
2assignvariableop_79_adam_separable_conv2d_4_bias_v:1D
*assignvariableop_80_adam_conv2d_2_kernel_v:16
(assignvariableop_81_adam_conv2d_2_bias_v:
identity_83’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_80’AssignVariableOp_81’AssignVariableOp_9Υ&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*α%
valueΧ%BΤ%SB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names·
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*»
value±B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesΝ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*β
_output_shapesΟ
Μ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	2
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

Identity_3’
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ͺ
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5₯
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

Identity_7Ή
AssignVariableOp_7AssignVariableOp4assignvariableop_7_separable_conv2d_depthwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ή
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
Identity_10Ώ
AssignVariableOp_10AssignVariableOp7assignvariableop_10_separable_conv2d_1_depthwise_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ώ
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
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¦
AssignVariableOp_16AssignVariableOpassignvariableop_16_dense_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ͺ
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¨
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ώ
AssignVariableOp_19AssignVariableOp7assignvariableop_19_separable_conv2d_2_depthwise_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ώ
AssignVariableOp_20AssignVariableOp7assignvariableop_20_separable_conv2d_2_pointwise_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_separable_conv2d_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ώ
AssignVariableOp_22AssignVariableOp7assignvariableop_22_separable_conv2d_3_depthwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ώ
AssignVariableOp_23AssignVariableOp7assignvariableop_23_separable_conv2d_3_pointwise_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24³
AssignVariableOp_24AssignVariableOp+assignvariableop_24_separable_conv2d_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ώ
AssignVariableOp_25AssignVariableOp7assignvariableop_25_separable_conv2d_4_depthwise_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ώ
AssignVariableOp_26AssignVariableOp7assignvariableop_26_separable_conv2d_4_pointwise_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_separable_conv2d_4_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28«
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29©
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30‘
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31‘
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_conv2d_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Δ
AssignVariableOp_34AssignVariableOp<assignvariableop_34_adam_separable_conv2d_depthwise_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Δ
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adam_separable_conv2d_pointwise_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Έ
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_separable_conv2d_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ζ
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_separable_conv2d_1_depthwise_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ζ
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_separable_conv2d_1_pointwise_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ί
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_separable_conv2d_1_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40²
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_1_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_conv2d_1_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42―
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43­
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adam_dense_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_1_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45―
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_1_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ζ
AssignVariableOp_46AssignVariableOp>assignvariableop_46_adam_separable_conv2d_2_depthwise_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ζ
AssignVariableOp_47AssignVariableOp>assignvariableop_47_adam_separable_conv2d_2_pointwise_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ί
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_separable_conv2d_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ζ
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_separable_conv2d_3_depthwise_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ζ
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adam_separable_conv2d_3_pointwise_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ί
AssignVariableOp_51AssignVariableOp2assignvariableop_51_adam_separable_conv2d_3_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ζ
AssignVariableOp_52AssignVariableOp>assignvariableop_52_adam_separable_conv2d_4_depthwise_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ζ
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_separable_conv2d_4_pointwise_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ί
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_separable_conv2d_4_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56°
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57°
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_conv2d_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_conv2d_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Δ
AssignVariableOp_59AssignVariableOp<assignvariableop_59_adam_separable_conv2d_depthwise_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Δ
AssignVariableOp_60AssignVariableOp<assignvariableop_60_adam_separable_conv2d_pointwise_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Έ
AssignVariableOp_61AssignVariableOp0assignvariableop_61_adam_separable_conv2d_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ζ
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_separable_conv2d_1_depthwise_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ζ
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adam_separable_conv2d_1_pointwise_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ί
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_separable_conv2d_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65²
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_1_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66°
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_1_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67―
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_dense_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68­
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_dense_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69±
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70―
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ζ
AssignVariableOp_71AssignVariableOp>assignvariableop_71_adam_separable_conv2d_2_depthwise_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ζ
AssignVariableOp_72AssignVariableOp>assignvariableop_72_adam_separable_conv2d_2_pointwise_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ί
AssignVariableOp_73AssignVariableOp2assignvariableop_73_adam_separable_conv2d_2_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Ζ
AssignVariableOp_74AssignVariableOp>assignvariableop_74_adam_separable_conv2d_3_depthwise_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ζ
AssignVariableOp_75AssignVariableOp>assignvariableop_75_adam_separable_conv2d_3_pointwise_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ί
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_separable_conv2d_3_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Ζ
AssignVariableOp_77AssignVariableOp>assignvariableop_77_adam_separable_conv2d_4_depthwise_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Ζ
AssignVariableOp_78AssignVariableOp>assignvariableop_78_adam_separable_conv2d_4_pointwise_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Ί
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_separable_conv2d_4_bias_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80²
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_conv2d_2_kernel_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81°
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_conv2d_2_bias_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpκ
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82f
Identity_83IdentityIdentity_82:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_83?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_83Identity_83:output:0*»
_input_shapes©
¦: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Έϋ
Χ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_103042

inputsM
3mnist_enc_var_conv2d_conv2d_readvariableop_resource:B
4mnist_enc_var_conv2d_biasadd_readvariableop_resource:a
Gmnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_resource:c
Imnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_1_resource:1L
>mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource:1c
Imnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource:1e
Kmnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11N
@mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource:1O
5mnist_enc_var_conv2d_1_conv2d_readvariableop_resource:11D
6mnist_enc_var_conv2d_1_biasadd_readvariableop_resource:1D
2mnist_enc_var_dense_matmul_readvariableop_resource:11A
3mnist_enc_var_dense_biasadd_readvariableop_resource:1F
4mnist_enc_var_dense_1_matmul_readvariableop_resource:11C
5mnist_enc_var_dense_1_biasadd_readvariableop_resource:1c
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
identity’-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp’,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp’7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp’@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp’Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1’7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp’@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp’Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1’7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp’@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp’Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1’+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp’*mnist_enc_var/conv2d/Conv2D/ReadVariableOp’-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp’,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp’*mnist_enc_var/dense/BiasAdd/ReadVariableOp’)mnist_enc_var/dense/MatMul/ReadVariableOp’,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp’+mnist_enc_var/dense_1/MatMul/ReadVariableOp’5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp’>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp’@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1’7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp’@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp’Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Τ
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpReadVariableOp3mnist_enc_var_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpβ
mnist_enc_var/conv2d/Conv2DConv2Dinputs2mnist_enc_var/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_enc_var/conv2d/Conv2DΛ
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpReadVariableOp4mnist_enc_var_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpά
mnist_enc_var/conv2d/BiasAddBiasAdd$mnist_enc_var/conv2d/Conv2D:output:03mnist_enc_var/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
mnist_enc_var/conv2d/BiasAdd
mnist_enc_var/conv2d/SeluSelu%mnist_enc_var/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
mnist_enc_var/conv2d/Selu
>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpGmnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpImnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02B
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1Η
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            27
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeΟ
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateβ
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative'mnist_enc_var/conv2d/Selu:activations:0Fmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2;
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseέ
/mnist_enc_var/separable_conv2d/separable_conv2dConv2DBmnist_enc_var/separable_conv2d/separable_conv2d/depthwise:output:0Hmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
21
/mnist_enc_var/separable_conv2d/separable_conv2dι
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp>mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype027
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp
&mnist_enc_var/separable_conv2d/BiasAddBiasAdd8mnist_enc_var/separable_conv2d/separable_conv2d:output:0=mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12(
&mnist_enc_var/separable_conv2d/BiasAdd½
#mnist_enc_var/separable_conv2d/SeluSelu/mnist_enc_var/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12%
#mnist_enc_var/separable_conv2d/Selu
@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpImnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Λ
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeΣ
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateς
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative1mnist_enc_var/separable_conv2d/Selu:activations:0Hmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseε
1mnist_enc_var/separable_conv2d_1/separable_conv2dConv2DDmnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise:output:0Jmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_enc_var/separable_conv2d_1/separable_conv2dο
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp
(mnist_enc_var/separable_conv2d_1/BiasAddBiasAdd:mnist_enc_var/separable_conv2d_1/separable_conv2d:output:0?mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_enc_var/separable_conv2d_1/BiasAddΓ
%mnist_enc_var/separable_conv2d_1/SeluSelu1mnist_enc_var/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_enc_var/separable_conv2d_1/SeluΪ
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5mnist_enc_var_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02.
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp
mnist_enc_var/conv2d_1/Conv2DConv2D3mnist_enc_var/separable_conv2d_1/Selu:activations:04mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
mnist_enc_var/conv2d_1/Conv2DΡ
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6mnist_enc_var_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02/
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpδ
mnist_enc_var/conv2d_1/BiasAddBiasAdd&mnist_enc_var/conv2d_1/Conv2D:output:05mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12 
mnist_enc_var/conv2d_1/BiasAdd₯
mnist_enc_var/conv2d_1/SeluSelu'mnist_enc_var/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
mnist_enc_var/conv2d_1/SeluΟ
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indices
+mnist_enc_var/global_average_pooling2d/MeanMean)mnist_enc_var/conv2d_1/Selu:activations:0Fmnist_enc_var/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12-
+mnist_enc_var/global_average_pooling2d/MeanΙ
)mnist_enc_var/dense/MatMul/ReadVariableOpReadVariableOp2mnist_enc_var_dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02+
)mnist_enc_var/dense/MatMul/ReadVariableOpέ
mnist_enc_var/dense/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:01mnist_enc_var/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense/MatMulΘ
*mnist_enc_var/dense/BiasAdd/ReadVariableOpReadVariableOp3mnist_enc_var_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02,
*mnist_enc_var/dense/BiasAdd/ReadVariableOpΡ
mnist_enc_var/dense/BiasAddBiasAdd$mnist_enc_var/dense/MatMul:product:02mnist_enc_var/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense/BiasAdd
mnist_enc_var/dense/SeluSelu$mnist_enc_var/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense/SeluΟ
+mnist_enc_var/dense_1/MatMul/ReadVariableOpReadVariableOp4mnist_enc_var_dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02-
+mnist_enc_var/dense_1/MatMul/ReadVariableOpγ
mnist_enc_var/dense_1/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:03mnist_enc_var/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense_1/MatMulΞ
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpReadVariableOp5mnist_enc_var_dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02.
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpΩ
mnist_enc_var/dense_1/BiasAddBiasAdd&mnist_enc_var/dense_1/MatMul:product:04mnist_enc_var/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense_1/BiasAdd
mnist_enc_var/dense_1/SeluSelu&mnist_enc_var/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense_1/Selu
mnist_enc_var/sampling/ShapeShape(mnist_enc_var/dense_1/Selu:activations:0*
T0*
_output_shapes
:2
mnist_enc_var/sampling/Shape
)mnist_enc_var/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mnist_enc_var/sampling/random_normal/mean
+mnist_enc_var/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+mnist_enc_var/sampling/random_normal/stddev
9mnist_enc_var/sampling/random_normal/RandomStandardNormalRandomStandardNormal%mnist_enc_var/sampling/Shape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2ψ»΄2;
9mnist_enc_var/sampling/random_normal/RandomStandardNormal
(mnist_enc_var/sampling/random_normal/mulMulBmnist_enc_var/sampling/random_normal/RandomStandardNormal:output:04mnist_enc_var/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????12*
(mnist_enc_var/sampling/random_normal/mulι
$mnist_enc_var/sampling/random_normalAddV2,mnist_enc_var/sampling/random_normal/mul:z:02mnist_enc_var/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????12&
$mnist_enc_var/sampling/random_normal
 mnist_enc_var/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 mnist_enc_var/sampling/truediv/y?
mnist_enc_var/sampling/truedivRealDiv(mnist_enc_var/dense_1/Selu:activations:0)mnist_enc_var/sampling/truediv/y:output:0*
T0*'
_output_shapes
:?????????12 
mnist_enc_var/sampling/truediv
mnist_enc_var/sampling/ExpExp"mnist_enc_var/sampling/truediv:z:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/sampling/Exp»
mnist_enc_var/sampling/mulMul(mnist_enc_var/sampling/random_normal:z:0mnist_enc_var/sampling/Exp:y:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/sampling/mul»
mnist_enc_var/sampling/addAddV2mnist_enc_var/sampling/mul:z:0&mnist_enc_var/dense/Selu:activations:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/sampling/add
mnist_dec_var/reshape/ShapeShapemnist_enc_var/sampling/add:z:0*
T0*
_output_shapes
:2
mnist_dec_var/reshape/Shape 
)mnist_dec_var/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)mnist_dec_var/reshape/strided_slice/stack€
+mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_1€
+mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_2ζ
#mnist_dec_var/reshape/strided_sliceStridedSlice$mnist_dec_var/reshape/Shape:output:02mnist_dec_var/reshape/strided_slice/stack:output:04mnist_dec_var/reshape/strided_slice/stack_1:output:04mnist_dec_var/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#mnist_dec_var/reshape/strided_slice
%mnist_dec_var/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/1
%mnist_dec_var/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/2
%mnist_dec_var/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/3Ύ
#mnist_dec_var/reshape/Reshape/shapePack,mnist_dec_var/reshape/strided_slice:output:0.mnist_dec_var/reshape/Reshape/shape/1:output:0.mnist_dec_var/reshape/Reshape/shape/2:output:0.mnist_dec_var/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#mnist_dec_var/reshape/Reshape/shapeΡ
mnist_dec_var/reshape/ReshapeReshapemnist_enc_var/sampling/add:z:0,mnist_dec_var/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
mnist_dec_var/reshape/Reshape
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02B
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02D
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Λ
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeΣ
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateη
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative&mnist_dec_var/reshape/Reshape:output:0Hmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseε
1mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_2/separable_conv2dο
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_2/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_2/BiasAddΓ
%mnist_dec_var/separable_conv2d_2/SeluSelu1mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_2/Selu
!mnist_dec_var/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!mnist_dec_var/up_sampling2d/Const
#mnist_dec_var/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#mnist_dec_var/up_sampling2d/Const_1Θ
mnist_dec_var/up_sampling2d/mulMul*mnist_dec_var/up_sampling2d/Const:output:0,mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2!
mnist_dec_var/up_sampling2d/mul»
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_2/Selu:activations:0#mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2:
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Λ
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeΣ
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeImnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseε
1mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_3/separable_conv2dο
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_3/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_3/BiasAddΓ
%mnist_dec_var/separable_conv2d_3/SeluSelu1mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_3/Selu
#mnist_dec_var/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#mnist_dec_var/up_sampling2d_1/Const
%mnist_dec_var/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%mnist_dec_var/up_sampling2d_1/Const_1Π
!mnist_dec_var/up_sampling2d_1/mulMul,mnist_dec_var/up_sampling2d_1/Const:output:0.mnist_dec_var/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2#
!mnist_dec_var/up_sampling2d_1/mulΑ
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_3/Selu:activations:0%mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2<
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Λ
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeΣ
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeKmnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseε
1mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_4/separable_conv2dο
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_4/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_4/BiasAddΓ
%mnist_dec_var/separable_conv2d_4/SeluSelu1mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_4/SeluΪ
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02.
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp
mnist_dec_var/conv2d_2/Conv2DConv2D3mnist_dec_var/separable_conv2d_4/Selu:activations:04mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_dec_var/conv2d_2/Conv2DΡ
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpδ
mnist_dec_var/conv2d_2/BiasAddBiasAdd&mnist_dec_var/conv2d_2/Conv2D:output:05mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2 
mnist_dec_var/conv2d_2/BiasAdd₯
mnist_dec_var/conv2d_2/SeluSelu'mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
mnist_dec_var/conv2d_2/Selu
IdentityIdentity)mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityα
NoOpNoOp.^mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-^mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp8^mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1,^mnist_enc_var/conv2d/BiasAdd/ReadVariableOp+^mnist_enc_var/conv2d/Conv2D/ReadVariableOp.^mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp-^mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp+^mnist_enc_var/dense/BiasAdd/ReadVariableOp*^mnist_enc_var/dense/MatMul/ReadVariableOp-^mnist_enc_var/dense_1/BiasAdd/ReadVariableOp,^mnist_enc_var/dense_1/MatMul/ReadVariableOp6^mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp?^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpA^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_18^mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpA^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpC^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2^
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp2\
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp2r
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp2
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp2
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_12r
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp2
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp2
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_12r
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp2
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp2
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_12Z
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp2X
*mnist_enc_var/conv2d/Conv2D/ReadVariableOp*mnist_enc_var/conv2d/Conv2D/ReadVariableOp2^
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp2\
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp2X
*mnist_enc_var/dense/BiasAdd/ReadVariableOp*mnist_enc_var/dense/BiasAdd/ReadVariableOp2V
)mnist_enc_var/dense/MatMul/ReadVariableOp)mnist_enc_var/dense/MatMul/ReadVariableOp2\
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp2Z
+mnist_enc_var/dense_1/MatMul/ReadVariableOp+mnist_enc_var/dense_1/MatMul/ReadVariableOp2n
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp2
>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp2
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_12r
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp2
@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp2
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs

_
C__inference_reshape_layer_call_and_return_conditional_losses_103551

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
strided_slice/stack_2β
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
Reshape/shape/3Ί
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
Έϋ
Χ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102912

inputsM
3mnist_enc_var_conv2d_conv2d_readvariableop_resource:B
4mnist_enc_var_conv2d_biasadd_readvariableop_resource:a
Gmnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_resource:c
Imnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_1_resource:1L
>mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource:1c
Imnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource:1e
Kmnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:11N
@mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource:1O
5mnist_enc_var_conv2d_1_conv2d_readvariableop_resource:11D
6mnist_enc_var_conv2d_1_biasadd_readvariableop_resource:1D
2mnist_enc_var_dense_matmul_readvariableop_resource:11A
3mnist_enc_var_dense_biasadd_readvariableop_resource:1F
4mnist_enc_var_dense_1_matmul_readvariableop_resource:11C
5mnist_enc_var_dense_1_biasadd_readvariableop_resource:1c
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
identity’-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp’,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp’7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp’@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp’Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1’7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp’@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp’Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1’7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp’@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp’Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1’+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp’*mnist_enc_var/conv2d/Conv2D/ReadVariableOp’-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp’,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp’*mnist_enc_var/dense/BiasAdd/ReadVariableOp’)mnist_enc_var/dense/MatMul/ReadVariableOp’,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp’+mnist_enc_var/dense_1/MatMul/ReadVariableOp’5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp’>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp’@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1’7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp’@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp’Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Τ
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpReadVariableOp3mnist_enc_var_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpβ
mnist_enc_var/conv2d/Conv2DConv2Dinputs2mnist_enc_var/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_enc_var/conv2d/Conv2DΛ
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpReadVariableOp4mnist_enc_var_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpά
mnist_enc_var/conv2d/BiasAddBiasAdd$mnist_enc_var/conv2d/Conv2D:output:03mnist_enc_var/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
mnist_enc_var/conv2d/BiasAdd
mnist_enc_var/conv2d/SeluSelu%mnist_enc_var/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
mnist_enc_var/conv2d/Selu
>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpGmnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpImnist_enc_var_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02B
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1Η
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            27
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeΟ
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateβ
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative'mnist_enc_var/conv2d/Selu:activations:0Fmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2;
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseέ
/mnist_enc_var/separable_conv2d/separable_conv2dConv2DBmnist_enc_var/separable_conv2d/separable_conv2d/depthwise:output:0Hmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
21
/mnist_enc_var/separable_conv2d/separable_conv2dι
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp>mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype027
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp
&mnist_enc_var/separable_conv2d/BiasAddBiasAdd8mnist_enc_var/separable_conv2d/separable_conv2d:output:0=mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12(
&mnist_enc_var/separable_conv2d/BiasAdd½
#mnist_enc_var/separable_conv2d/SeluSelu/mnist_enc_var/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12%
#mnist_enc_var/separable_conv2d/Selu
@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpImnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Λ
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeΣ
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateς
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative1mnist_enc_var/separable_conv2d/Selu:activations:0Hmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseε
1mnist_enc_var/separable_conv2d_1/separable_conv2dConv2DDmnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise:output:0Jmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_enc_var/separable_conv2d_1/separable_conv2dο
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp
(mnist_enc_var/separable_conv2d_1/BiasAddBiasAdd:mnist_enc_var/separable_conv2d_1/separable_conv2d:output:0?mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_enc_var/separable_conv2d_1/BiasAddΓ
%mnist_enc_var/separable_conv2d_1/SeluSelu1mnist_enc_var/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_enc_var/separable_conv2d_1/SeluΪ
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5mnist_enc_var_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02.
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp
mnist_enc_var/conv2d_1/Conv2DConv2D3mnist_enc_var/separable_conv2d_1/Selu:activations:04mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2
mnist_enc_var/conv2d_1/Conv2DΡ
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6mnist_enc_var_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02/
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpδ
mnist_enc_var/conv2d_1/BiasAddBiasAdd&mnist_enc_var/conv2d_1/Conv2D:output:05mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12 
mnist_enc_var/conv2d_1/BiasAdd₯
mnist_enc_var/conv2d_1/SeluSelu'mnist_enc_var/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
mnist_enc_var/conv2d_1/SeluΟ
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indices
+mnist_enc_var/global_average_pooling2d/MeanMean)mnist_enc_var/conv2d_1/Selu:activations:0Fmnist_enc_var/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12-
+mnist_enc_var/global_average_pooling2d/MeanΙ
)mnist_enc_var/dense/MatMul/ReadVariableOpReadVariableOp2mnist_enc_var_dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02+
)mnist_enc_var/dense/MatMul/ReadVariableOpέ
mnist_enc_var/dense/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:01mnist_enc_var/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense/MatMulΘ
*mnist_enc_var/dense/BiasAdd/ReadVariableOpReadVariableOp3mnist_enc_var_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02,
*mnist_enc_var/dense/BiasAdd/ReadVariableOpΡ
mnist_enc_var/dense/BiasAddBiasAdd$mnist_enc_var/dense/MatMul:product:02mnist_enc_var/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense/BiasAdd
mnist_enc_var/dense/SeluSelu$mnist_enc_var/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense/SeluΟ
+mnist_enc_var/dense_1/MatMul/ReadVariableOpReadVariableOp4mnist_enc_var_dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02-
+mnist_enc_var/dense_1/MatMul/ReadVariableOpγ
mnist_enc_var/dense_1/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:03mnist_enc_var/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense_1/MatMulΞ
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpReadVariableOp5mnist_enc_var_dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02.
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpΩ
mnist_enc_var/dense_1/BiasAddBiasAdd&mnist_enc_var/dense_1/MatMul:product:04mnist_enc_var/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense_1/BiasAdd
mnist_enc_var/dense_1/SeluSelu&mnist_enc_var/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/dense_1/Selu
mnist_enc_var/sampling/ShapeShape(mnist_enc_var/dense_1/Selu:activations:0*
T0*
_output_shapes
:2
mnist_enc_var/sampling/Shape
)mnist_enc_var/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)mnist_enc_var/sampling/random_normal/mean
+mnist_enc_var/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+mnist_enc_var/sampling/random_normal/stddev
9mnist_enc_var/sampling/random_normal/RandomStandardNormalRandomStandardNormal%mnist_enc_var/sampling/Shape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2 ?Σ2;
9mnist_enc_var/sampling/random_normal/RandomStandardNormal
(mnist_enc_var/sampling/random_normal/mulMulBmnist_enc_var/sampling/random_normal/RandomStandardNormal:output:04mnist_enc_var/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????12*
(mnist_enc_var/sampling/random_normal/mulι
$mnist_enc_var/sampling/random_normalAddV2,mnist_enc_var/sampling/random_normal/mul:z:02mnist_enc_var/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????12&
$mnist_enc_var/sampling/random_normal
 mnist_enc_var/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 mnist_enc_var/sampling/truediv/y?
mnist_enc_var/sampling/truedivRealDiv(mnist_enc_var/dense_1/Selu:activations:0)mnist_enc_var/sampling/truediv/y:output:0*
T0*'
_output_shapes
:?????????12 
mnist_enc_var/sampling/truediv
mnist_enc_var/sampling/ExpExp"mnist_enc_var/sampling/truediv:z:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/sampling/Exp»
mnist_enc_var/sampling/mulMul(mnist_enc_var/sampling/random_normal:z:0mnist_enc_var/sampling/Exp:y:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/sampling/mul»
mnist_enc_var/sampling/addAddV2mnist_enc_var/sampling/mul:z:0&mnist_enc_var/dense/Selu:activations:0*
T0*'
_output_shapes
:?????????12
mnist_enc_var/sampling/add
mnist_dec_var/reshape/ShapeShapemnist_enc_var/sampling/add:z:0*
T0*
_output_shapes
:2
mnist_dec_var/reshape/Shape 
)mnist_dec_var/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)mnist_dec_var/reshape/strided_slice/stack€
+mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_1€
+mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_2ζ
#mnist_dec_var/reshape/strided_sliceStridedSlice$mnist_dec_var/reshape/Shape:output:02mnist_dec_var/reshape/strided_slice/stack:output:04mnist_dec_var/reshape/strided_slice/stack_1:output:04mnist_dec_var/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#mnist_dec_var/reshape/strided_slice
%mnist_dec_var/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/1
%mnist_dec_var/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/2
%mnist_dec_var/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%mnist_dec_var/reshape/Reshape/shape/3Ύ
#mnist_dec_var/reshape/Reshape/shapePack,mnist_dec_var/reshape/strided_slice:output:0.mnist_dec_var/reshape/Reshape/shape/1:output:0.mnist_dec_var/reshape/Reshape/shape/2:output:0.mnist_dec_var/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#mnist_dec_var/reshape/Reshape/shapeΡ
mnist_dec_var/reshape/ReshapeReshapemnist_enc_var/sampling/add:z:0,mnist_dec_var/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
mnist_dec_var/reshape/Reshape
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02B
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02D
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Λ
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeΣ
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateη
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative&mnist_dec_var/reshape/Reshape:output:0Hmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseε
1mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_2/separable_conv2dο
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_2/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_2/BiasAddΓ
%mnist_dec_var/separable_conv2d_2/SeluSelu1mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_2/Selu
!mnist_dec_var/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!mnist_dec_var/up_sampling2d/Const
#mnist_dec_var/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#mnist_dec_var/up_sampling2d/Const_1Θ
mnist_dec_var/up_sampling2d/mulMul*mnist_dec_var/up_sampling2d/Const:output:0,mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2!
mnist_dec_var/up_sampling2d/mul»
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_2/Selu:activations:0#mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2:
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Λ
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeΣ
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeImnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseε
1mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_3/separable_conv2dο
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_3/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_3/BiasAddΓ
%mnist_dec_var/separable_conv2d_3/SeluSelu1mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_3/Selu
#mnist_dec_var/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#mnist_dec_var/up_sampling2d_1/Const
%mnist_dec_var/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%mnist_dec_var/up_sampling2d_1/Const_1Π
!mnist_dec_var/up_sampling2d_1/mulMul,mnist_dec_var/up_sampling2d_1/Const:output:0.mnist_dec_var/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2#
!mnist_dec_var/up_sampling2d_1/mulΑ
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_3/Selu:activations:0%mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2<
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpImnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02B
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpKmnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02D
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Λ
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeΣ
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeKmnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseε
1mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_4/separable_conv2dο
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_4/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12*
(mnist_dec_var/separable_conv2d_4/BiasAddΓ
%mnist_dec_var/separable_conv2d_4/SeluSelu1mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12'
%mnist_dec_var/separable_conv2d_4/SeluΪ
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02.
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp
mnist_dec_var/conv2d_2/Conv2DConv2D3mnist_dec_var/separable_conv2d_4/Selu:activations:04mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_dec_var/conv2d_2/Conv2DΡ
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpδ
mnist_dec_var/conv2d_2/BiasAddBiasAdd&mnist_dec_var/conv2d_2/Conv2D:output:05mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2 
mnist_dec_var/conv2d_2/BiasAdd₯
mnist_dec_var/conv2d_2/SeluSelu'mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
mnist_dec_var/conv2d_2/Selu
IdentityIdentity)mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityα
NoOpNoOp.^mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-^mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp8^mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1,^mnist_enc_var/conv2d/BiasAdd/ReadVariableOp+^mnist_enc_var/conv2d/Conv2D/ReadVariableOp.^mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp-^mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp+^mnist_enc_var/dense/BiasAdd/ReadVariableOp*^mnist_enc_var/dense/MatMul/ReadVariableOp-^mnist_enc_var/dense_1/BiasAdd/ReadVariableOp,^mnist_enc_var/dense_1/MatMul/ReadVariableOp6^mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp?^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpA^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_18^mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpA^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpC^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2^
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp2\
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp2r
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp2
@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp2
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_12r
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp2
@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp2
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_12r
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp2
@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp2
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_12Z
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp2X
*mnist_enc_var/conv2d/Conv2D/ReadVariableOp*mnist_enc_var/conv2d/Conv2D/ReadVariableOp2^
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp2\
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp2X
*mnist_enc_var/dense/BiasAdd/ReadVariableOp*mnist_enc_var/dense/BiasAdd/ReadVariableOp2V
)mnist_enc_var/dense/MatMul/ReadVariableOp)mnist_enc_var/dense/MatMul/ReadVariableOp2\
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp2Z
+mnist_enc_var/dense_1/MatMul/ReadVariableOp+mnist_enc_var/dense_1/MatMul/ReadVariableOp2n
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp2
>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp2
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_12r
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp2
@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp2
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs


)__inference_conv2d_1_layer_call_fn_103459

inputs!
unknown:11
	unknown_0:1
identity’StatefulPartitionedCall?
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1013962
StatefulPartitionedCall
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
Φ
J
.__inference_up_sampling2d_layer_call_fn_101801

inputs
identityν
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
GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1017952
PartitionedCall
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
Ν
D
(__inference_reshape_layer_call_fn_103537

inputs
identityΜ
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
GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1018992
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
ΐ

.__inference_mnist_enc_var_layer_call_fn_103108

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1016092
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs

q
D__inference_sampling_layer_call_and_return_conditional_losses_101453

inputs
inputs_1
identityF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevΞ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2¦σσ2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????12
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????12
random_normal[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ym
truedivRealDivinputs_1truediv/y:output:0*
T0*'
_output_shapes
:?????????12	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????12
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????12
mulV
addAddV2mul:z:0inputs*
T0*'
_output_shapes
:?????????12
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????1:?????????1:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
κ
ύ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_101396

inputs8
conv2d_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
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
ΐ

.__inference_mnist_enc_var_layer_call_fn_103075

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1014562
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ω'

I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_101942

inputs3
separable_conv2d_2_101901:3
separable_conv2d_2_101903:1'
separable_conv2d_2_101905:13
separable_conv2d_3_101909:13
separable_conv2d_3_101911:11'
separable_conv2d_3_101913:13
separable_conv2d_4_101917:13
separable_conv2d_4_101919:11'
separable_conv2d_4_101921:1)
conv2d_2_101936:1
conv2d_2_101938:
identity’ conv2d_2/StatefulPartitionedCall’*separable_conv2d_2/StatefulPartitionedCall’*separable_conv2d_3/StatefulPartitionedCall’*separable_conv2d_4/StatefulPartitionedCallά
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
GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1018992
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_101901separable_conv2d_2_101903separable_conv2d_2_101905*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1017702,
*separable_conv2d_2/StatefulPartitionedCall­
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
GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1017952
up_sampling2d/PartitionedCall 
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_101909separable_conv2d_3_101911separable_conv2d_3_101913*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1018182,
*separable_conv2d_3/StatefulPartitionedCall³
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
GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1018432!
up_sampling2d_1/PartitionedCall’
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_101917separable_conv2d_4_101919separable_conv2d_4_101921*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1018662,
*separable_conv2d_4/StatefulPartitionedCallή
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_101936conv2d_2_101938*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1019352"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityψ
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
₯

L__inference_separable_conv2d_layer_call_and_return_conditional_losses_101293

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’separable_conv2d/ReadVariableOp’!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOpΉ
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
separable_conv2d/dilation_rateφ
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwiseσ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity­
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
δ
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_101341

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
·
U
9__inference_global_average_pooling2d_layer_call_fn_101347

inputs
identityή
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1013412
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


-__inference_mnist_ae_var_layer_call_fn_102727

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1$

unknown_13:$

unknown_14:1

unknown_15:1$

unknown_16:1$

unknown_17:11

unknown_18:1$

unknown_19:1$

unknown_20:11

unknown_21:1$

unknown_22:1

unknown_23:
identity’StatefulPartitionedCallΞ
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
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_1022232
StatefulPartitionedCall
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

ς
A__inference_dense_layer_call_and_return_conditional_losses_101414

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
°
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_101795

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
strided_slice/stack_2Ξ
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
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
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
λ
Μ
.__inference_mnist_dec_var_layer_call_fn_101967

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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1019422
StatefulPartitionedCall
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
Κ-

I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101456

inputs'
conv2d_101366:
conv2d_101368:1
separable_conv2d_101371:1
separable_conv2d_101373:1%
separable_conv2d_101375:13
separable_conv2d_1_101378:13
separable_conv2d_1_101380:11'
separable_conv2d_1_101382:1)
conv2d_1_101397:11
conv2d_1_101399:1
dense_101415:11
dense_101417:1 
dense_1_101432:11
dense_1_101434:1
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’(separable_conv2d/StatefulPartitionedCall’*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_101366conv2d_101368*
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
GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1013652 
conv2d/StatefulPartitionedCall
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_101371separable_conv2d_101373separable_conv2d_101375*
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
GPU2*0J 8 *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_1012932*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_101378separable_conv2d_1_101380separable_conv2d_1_101382*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1013222,
*separable_conv2d_1/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_101397conv2d_1_101399*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1013962"
 conv2d_1/StatefulPartitionedCallͺ
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
GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1013412*
(global_average_pooling2d/PartitionedCall³
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_101415dense_101417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1014142
dense/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_101432dense_1_101434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1014312!
dense_1/StatefulPartitionedCallΊ
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1014532"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????12

IdentityΟ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ΐ

.__inference_mnist_enc_var_layer_call_fn_101673

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallenc_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1016092
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
ν
ί	
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102223

inputs.
mnist_enc_var_102170:"
mnist_enc_var_102172:.
mnist_enc_var_102174:.
mnist_enc_var_102176:1"
mnist_enc_var_102178:1.
mnist_enc_var_102180:1.
mnist_enc_var_102182:11"
mnist_enc_var_102184:1.
mnist_enc_var_102186:11"
mnist_enc_var_102188:1&
mnist_enc_var_102190:11"
mnist_enc_var_102192:1&
mnist_enc_var_102194:11"
mnist_enc_var_102196:1.
mnist_dec_var_102199:.
mnist_dec_var_102201:1"
mnist_dec_var_102203:1.
mnist_dec_var_102205:1.
mnist_dec_var_102207:11"
mnist_dec_var_102209:1.
mnist_dec_var_102211:1.
mnist_dec_var_102213:11"
mnist_dec_var_102215:1.
mnist_dec_var_102217:1"
mnist_dec_var_102219:
identity’%mnist_dec_var/StatefulPartitionedCall’%mnist_enc_var/StatefulPartitionedCallΠ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallinputsmnist_enc_var_102170mnist_enc_var_102172mnist_enc_var_102174mnist_enc_var_102176mnist_enc_var_102178mnist_enc_var_102180mnist_enc_var_102182mnist_enc_var_102184mnist_enc_var_102186mnist_enc_var_102188mnist_enc_var_102190mnist_enc_var_102192mnist_enc_var_102194mnist_enc_var_102196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_1014562'
%mnist_enc_var/StatefulPartitionedCallΚ
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_102199mnist_dec_var_102201mnist_dec_var_102203mnist_dec_var_102205mnist_dec_var_102207mnist_dec_var_102209mnist_dec_var_102211mnist_dec_var_102213mnist_dec_var_102215mnist_dec_var_102217mnist_dec_var_102219*
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1019422'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs

τ
C__inference_dense_1_layer_call_and_return_conditional_losses_101431

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
§

N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_101866

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:11-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’separable_conv2d/ReadVariableOp’!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02!
separable_conv2d/ReadVariableOpΉ
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
separable_conv2d/dilation_rateφ
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingSAME*
strides
2
separable_conv2d/depthwiseσ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity­
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
λ
Μ
.__inference_mnist_dec_var_layer_call_fn_103300

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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1020452
StatefulPartitionedCall
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


-__inference_mnist_ae_var_layer_call_fn_102782

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1$

unknown_13:$

unknown_14:1

unknown_15:1$

unknown_16:1$

unknown_17:11

unknown_18:1$

unknown_19:1$

unknown_20:11

unknown_21:1$

unknown_22:1

unknown_23:
identity’StatefulPartitionedCallΞ
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
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_1023892
StatefulPartitionedCall
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
²
g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_101843

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
strided_slice/stack_2Ξ
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
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
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


-__inference_mnist_ae_var_layer_call_fn_102276

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
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1$

unknown_13:$

unknown_14:1

unknown_15:1$

unknown_16:1$

unknown_17:11

unknown_18:1$

unknown_19:1$

unknown_20:11

unknown_21:1$

unknown_22:1

unknown_23:
identity’StatefulPartitionedCallΞ
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
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_1022232
StatefulPartitionedCall
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameenc_in
?₯
'
__inference__traced_save_103840
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
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableopB
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
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopI
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
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopI
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

identity_1’MergeV2Checkpoints
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
ShardedFilenameΟ&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*α%
valueΧ%BΤ%SB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*»
value±B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesΪ%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop7savev2_adam_separable_conv2d_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_2_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_3_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop7savev2_adam_separable_conv2d_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_2_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_3_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*Υ
_input_shapesΓ
ΐ: : : : : : ::::1:1:1:11:1:11:1:11:1:11:1::1:1:1:11:1:1:11:1:1:: : ::::1:1:1:11:1:11:1:11:1:11:1::1:1:1:11:1:1:11:1:1:::::1:1:1:11:1:11:1:11:1:11:1::1:1:1:11:1:1:11:1:1:: 2(
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
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:,(
&
_output_shapes
::,(
&
_output_shapes
:1: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
:1:,(
&
_output_shapes
:11: 

_output_shapes
:1:,(
&
_output_shapes
:1: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :,!(
&
_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
::,$(
&
_output_shapes
:1: %

_output_shapes
:1:,&(
&
_output_shapes
:1:,'(
&
_output_shapes
:11: (

_output_shapes
:1:,)(
&
_output_shapes
:11: *

_output_shapes
:1:$+ 

_output_shapes

:11: ,

_output_shapes
:1:$- 

_output_shapes

:11: .

_output_shapes
:1:,/(
&
_output_shapes
::,0(
&
_output_shapes
:1: 1

_output_shapes
:1:,2(
&
_output_shapes
:1:,3(
&
_output_shapes
:11: 4

_output_shapes
:1:,5(
&
_output_shapes
:1:,6(
&
_output_shapes
:11: 7

_output_shapes
:1:,8(
&
_output_shapes
:1: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
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
:11: C

_output_shapes
:1:$D 

_output_shapes

:11: E

_output_shapes
:1:$F 

_output_shapes

:11: G

_output_shapes
:1:,H(
&
_output_shapes
::,I(
&
_output_shapes
:1: J

_output_shapes
:1:,K(
&
_output_shapes
:1:,L(
&
_output_shapes
:11: M

_output_shapes
:1:,N(
&
_output_shapes
:1:,O(
&
_output_shapes
:11: P

_output_shapes
:1:,Q(
&
_output_shapes
:1: R

_output_shapes
::S

_output_shapes
: 
§

N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_101770

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’separable_conv2d/ReadVariableOp’!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOpΉ
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
separable_conv2d/dilation_rateφ
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwiseσ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????1*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????12	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????12
Selu
IdentityIdentitySelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????12

Identity­
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
Ω'

I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_102130

dec_in3
separable_conv2d_2_102101:3
separable_conv2d_2_102103:1'
separable_conv2d_2_102105:13
separable_conv2d_3_102109:13
separable_conv2d_3_102111:11'
separable_conv2d_3_102113:13
separable_conv2d_4_102117:13
separable_conv2d_4_102119:11'
separable_conv2d_4_102121:1)
conv2d_2_102124:1
conv2d_2_102126:
identity’ conv2d_2/StatefulPartitionedCall’*separable_conv2d_2/StatefulPartitionedCall’*separable_conv2d_3/StatefulPartitionedCall’*separable_conv2d_4/StatefulPartitionedCallά
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
GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1018992
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_102101separable_conv2d_2_102103separable_conv2d_2_102105*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_1017702,
*separable_conv2d_2/StatefulPartitionedCall­
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
GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1017952
up_sampling2d/PartitionedCall 
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_102109separable_conv2d_3_102111separable_conv2d_3_102113*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_1018182,
*separable_conv2d_3/StatefulPartitionedCall³
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
GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1018432!
up_sampling2d_1/PartitionedCall’
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_102117separable_conv2d_4_102119separable_conv2d_4_102121*
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
GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_1018662,
*separable_conv2d_4/StatefulPartitionedCallή
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_102124conv2d_2_102126*
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
GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1019352"
 conv2d_2/StatefulPartitionedCall
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityψ
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
Όe
ΐ
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_103246

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
(conv2d_1_biasadd_readvariableop_resource:16
$dense_matmul_readvariableop_resource:113
%dense_biasadd_readvariableop_resource:18
&dense_1_matmul_readvariableop_resource:115
'dense_1_biasadd_readvariableop_resource:1
identity’conv2d/BiasAdd/ReadVariableOp’conv2d/Conv2D/ReadVariableOp’conv2d_1/BiasAdd/ReadVariableOp’conv2d_1/Conv2D/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’'separable_conv2d/BiasAdd/ReadVariableOp’0separable_conv2d/separable_conv2d/ReadVariableOp’2separable_conv2d/separable_conv2d/ReadVariableOp_1’)separable_conv2d_1/BiasAdd/ReadVariableOp’2separable_conv2d_1/separable_conv2d/ReadVariableOp’4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ͺ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpΈ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D‘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp€
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Seluζ
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpμ
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
/separable_conv2d/separable_conv2d/dilation_rateͺ
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise₯
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2dΏ
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOpΦ
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/BiasAdd
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d/Seluμ
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpς
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_1/separable_conv2d/dilation_rateΊ
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwise­
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2dΕ
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOpή
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/BiasAdd
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_1/Selu°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02 
conv2d_1/Conv2D/ReadVariableOpέ
conv2d_1/Conv2DConv2D%separable_conv2d_1/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
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
:?????????12
conv2d_1/BiasAdd{
conv2d_1/SeluSeluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
conv2d_1/Selu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesΟ
global_average_pooling2d/MeanMeanconv2d_1/Selu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????12
global_average_pooling2d/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense/MatMul/ReadVariableOp₯
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense/BiasAddj

dense/SeluSeludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12

dense/Selu₯
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense_1/MatMul/ReadVariableOp«
dense_1/MatMulMatMul&global_average_pooling2d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense_1/MatMul€
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02 
dense_1/BiasAdd/ReadVariableOp‘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
dense_1/BiasAddp
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????12
dense_1/Seluj
sampling/ShapeShapedense_1/Selu:activations:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sampling/random_normal/stddevι
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:?????????1*
dtype0*
seed±?ε)*
seed2­ηξ2-
+sampling/random_normal/RandomStandardNormalΟ
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????12
sampling/random_normal/mul±
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????12
sampling/random_normalm
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
sampling/truediv/y
sampling/truedivRealDivdense_1/Selu:activations:0sampling/truediv/y:output:0*
T0*'
_output_shapes
:?????????12
sampling/truedivk
sampling/ExpExpsampling/truediv:z:0*
T0*'
_output_shapes
:?????????12
sampling/Exp
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*'
_output_shapes
:?????????12
sampling/mul
sampling/addAddV2sampling/mul:z:0dense/Selu:activations:0*
T0*'
_output_shapes
:?????????12
sampling/addk
IdentityIdentitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:?????????12

Identityψ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
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
ς

(__inference_dense_1_layer_call_fn_103499

inputs
unknown:11
	unknown_0:1
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1014312
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
λ
Μ
.__inference_mnist_dec_var_layer_call_fn_103273

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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *R
fMRK
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_1019422
StatefulPartitionedCall
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
Υh
Δ
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_103365

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
identity’conv2d_2/BiasAdd/ReadVariableOp’conv2d_2/Conv2D/ReadVariableOp’)separable_conv2d_2/BiasAdd/ReadVariableOp’2separable_conv2d_2/separable_conv2d/ReadVariableOp’4separable_conv2d_2/separable_conv2d/ReadVariableOp_1’)separable_conv2d_3/BiasAdd/ReadVariableOp’2separable_conv2d_3/separable_conv2d/ReadVariableOp’4separable_conv2d_3/separable_conv2d/ReadVariableOp_1’)separable_conv2d_4/BiasAdd/ReadVariableOp’2separable_conv2d_4/separable_conv2d/ReadVariableOp’4separable_conv2d_4/separable_conv2d/ReadVariableOp_1T
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
reshape/Reshape/shape/3κ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapeμ
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpς
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_2/separable_conv2d/dilation_rate―
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativereshape/Reshape:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwise­
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2dΕ
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOpή
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_2/BiasAdd
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
up_sampling2d/Const_1
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_2/Selu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????1*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborμ
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpς
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_3/separable_conv2d/dilation_rate?
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise­
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2dΕ
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpή
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_3/BiasAdd
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
:?????????1*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborμ
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpς
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1―
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
1separable_conv2d_4/separable_conv2d/dilation_rateΤ
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????1*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise­
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????1*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2dΕ
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpή
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/BiasAdd
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????12
separable_conv2d_4/Selu°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02 
conv2d_2/Conv2D/ReadVariableOpέ
conv2d_2/Conv2DConv2D%separable_conv2d_4/Selu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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

IdentityΩ
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

_
C__inference_reshape_layer_call_and_return_conditional_losses_101899

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
strided_slice/stack_2β
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
Reshape/shape/3Ί
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
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ύ
serving_defaultͺ
A
enc_in7
serving_default_enc_in:0?????????I
mnist_dec_var8
StatefulPartitionedCall:0?????????tensorflow/serving/predict:υΚ
Α
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
Ί
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
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_network
ω
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	variables
trainable_variables
 regularization_losses
!	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_network
Υ
"iter

#beta_1

$beta_2
	%decay
&learning_rate'mΫ(mά)mέ*mή+mί,mΰ-mα.mβ/mγ0mδ1mε2mζ3mη4mθ5mι6mκ7mλ8mμ9mν:mξ;mο<mπ=mρ>mς?mσ'vτ(vυ)vφ*vχ+vψ,vω-vϊ.vϋ/vό0vύ1vώ2v?3v4v5v6v7v8v9v:v;v<v=v>v?v"
tf_deprecated_optimizer
ή
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24"
trackable_list_wrapper
ή
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24"
trackable_list_wrapper
 "
trackable_list_wrapper
Ξ
	variables
@layer_metrics
Alayer_regularization_losses

Blayers
trainable_variables
regularization_losses
Cnon_trainable_variables
Dmetrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
½

'kernel
(bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
έ
)depthwise_kernel
*pointwise_kernel
+bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
έ
,depthwise_kernel
-pointwise_kernel
.bias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

/kernel
0bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

1kernel
2bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
½

3kernel
4bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"
_tf_keras_layer
§
a	variables
btrainable_variables
cregularization_losses
d	keras_api
£__call__
+€&call_and_return_all_conditional_losses"
_tf_keras_layer

'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413"
trackable_list_wrapper

'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413"
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
elayer_metrics
flayer_regularization_losses

glayers
trainable_variables
regularization_losses
hnon_trainable_variables
imetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
§
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
₯__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
έ
5depthwise_kernel
6pointwise_kernel
7bias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
§
r	variables
strainable_variables
tregularization_losses
u	keras_api
©__call__
+ͺ&call_and_return_all_conditional_losses"
_tf_keras_layer
έ
8depthwise_kernel
9pointwise_kernel
:bias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
§
z	variables
{trainable_variables
|regularization_losses
}	keras_api
­__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
ί
;depthwise_kernel
<pointwise_kernel
=bias
~	variables
trainable_variables
regularization_losses
	keras_api
―__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

>kernel
?bias
	variables
trainable_variables
regularization_losses
	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10"
trackable_list_wrapper
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
	variables
layer_metrics
 layer_regularization_losses
layers
trainable_variables
 regularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
:112dense/kernel
:12
dense/bias
 :112dense_1/kernel
:12dense_1/bias
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
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
E	variables
layer_metrics
 layer_regularization_losses
layers
Ftrainable_variables
Gregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
I	variables
layer_metrics
 layer_regularization_losses
layers
Jtrainable_variables
Kregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
M	variables
layer_metrics
 layer_regularization_losses
layers
Ntrainable_variables
Oregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
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
΅
Q	variables
layer_metrics
 layer_regularization_losses
layers
Rtrainable_variables
Sregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
U	variables
 layer_metrics
 ‘layer_regularization_losses
’layers
Vtrainable_variables
Wregularization_losses
£non_trainable_variables
€metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Y	variables
₯layer_metrics
 ¦layer_regularization_losses
§layers
Ztrainable_variables
[regularization_losses
¨non_trainable_variables
©metrics
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
]	variables
ͺlayer_metrics
 «layer_regularization_losses
¬layers
^trainable_variables
_regularization_losses
­non_trainable_variables
?metrics
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
a	variables
―layer_metrics
 °layer_regularization_losses
±layers
btrainable_variables
cregularization_losses
²non_trainable_variables
³metrics
£__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0

1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
j	variables
΄layer_metrics
 ΅layer_regularization_losses
Άlayers
ktrainable_variables
lregularization_losses
·non_trainable_variables
Έmetrics
₯__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
n	variables
Ήlayer_metrics
 Ίlayer_regularization_losses
»layers
otrainable_variables
pregularization_losses
Όnon_trainable_variables
½metrics
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
r	variables
Ύlayer_metrics
 Ώlayer_regularization_losses
ΐlayers
strainable_variables
tregularization_losses
Αnon_trainable_variables
Βmetrics
©__call__
+ͺ&call_and_return_all_conditional_losses
'ͺ"call_and_return_conditional_losses"
_generic_user_object
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
v	variables
Γlayer_metrics
 Δlayer_regularization_losses
Εlayers
wtrainable_variables
xregularization_losses
Ζnon_trainable_variables
Ηmetrics
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
z	variables
Θlayer_metrics
 Ιlayer_regularization_losses
Κlayers
{trainable_variables
|regularization_losses
Λnon_trainable_variables
Μmetrics
­__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
;0
<1
=2"
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ά
~	variables
Νlayer_metrics
 Ξlayer_regularization_losses
Οlayers
trainable_variables
regularization_losses
Πnon_trainable_variables
Ρmetrics
―__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
?layer_metrics
 Σlayer_regularization_losses
Τlayers
trainable_variables
regularization_losses
Υnon_trainable_variables
Φmetrics
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

Χtotal

Ψcount
Ω	variables
Ϊ	keras_api"
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
Χ0
Ψ1"
trackable_list_wrapper
.
Ω	variables"
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
#:!112Adam/dense/kernel/m
:12Adam/dense/bias/m
%:#112Adam/dense_1/kernel/m
:12Adam/dense_1/bias/m
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
#:!112Adam/dense/kernel/v
:12Adam/dense/bias/v
%:#112Adam/dense_1/kernel/v
:12Adam/dense_1/bias/v
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
2?
-__inference_mnist_ae_var_layer_call_fn_102276
-__inference_mnist_ae_var_layer_call_fn_102727
-__inference_mnist_ae_var_layer_call_fn_102782
-__inference_mnist_ae_var_layer_call_fn_102497ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ζ2γ
!__inference__wrapped_model_101276½
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
annotationsͺ *-’*
(%
enc_in?????????
ξ2λ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102912
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_103042
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102553
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102609ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_mnist_enc_var_layer_call_fn_101487
.__inference_mnist_enc_var_layer_call_fn_103075
.__inference_mnist_enc_var_layer_call_fn_103108
.__inference_mnist_enc_var_layer_call_fn_101673ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_103177
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_103246
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101713
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101753ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_mnist_dec_var_layer_call_fn_101967
.__inference_mnist_dec_var_layer_call_fn_103273
.__inference_mnist_dec_var_layer_call_fn_103300
.__inference_mnist_dec_var_layer_call_fn_102097ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_103365
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_103430
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_102130
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_102163ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΚBΗ
$__inference_signature_wrapper_102672enc_in"
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
annotationsͺ *
 
Ρ2Ξ
'__inference_conv2d_layer_call_fn_103439’
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
annotationsͺ *
 
μ2ι
B__inference_conv2d_layer_call_and_return_conditional_losses_103450’
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
annotationsͺ *
 
2
1__inference_separable_conv2d_layer_call_fn_101305Χ
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
annotationsͺ *7’4
2/+???????????????????????????
«2¨
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_101293Χ
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
annotationsͺ *7’4
2/+???????????????????????????
2
3__inference_separable_conv2d_1_layer_call_fn_101334Χ
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
annotationsͺ *7’4
2/+???????????????????????????1
­2ͺ
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_101322Χ
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
annotationsͺ *7’4
2/+???????????????????????????1
Σ2Π
)__inference_conv2d_1_layer_call_fn_103459’
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
annotationsͺ *
 
ξ2λ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_103470’
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
annotationsͺ *
 
‘2
9__inference_global_average_pooling2d_layer_call_fn_101347ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
Ό2Ή
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_101341ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
Π2Ν
&__inference_dense_layer_call_fn_103479’
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
annotationsͺ *
 
λ2θ
A__inference_dense_layer_call_and_return_conditional_losses_103490’
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
annotationsͺ *
 
?2Ο
(__inference_dense_1_layer_call_fn_103499’
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
annotationsͺ *
 
ν2κ
C__inference_dense_1_layer_call_and_return_conditional_losses_103510’
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
annotationsͺ *
 
Σ2Π
)__inference_sampling_layer_call_fn_103516’
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
annotationsͺ *
 
ξ2λ
D__inference_sampling_layer_call_and_return_conditional_losses_103532’
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
annotationsͺ *
 
?2Ο
(__inference_reshape_layer_call_fn_103537’
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
annotationsͺ *
 
ν2κ
C__inference_reshape_layer_call_and_return_conditional_losses_103551’
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
annotationsͺ *
 
2
3__inference_separable_conv2d_2_layer_call_fn_101782Χ
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
annotationsͺ *7’4
2/+???????????????????????????
­2ͺ
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_101770Χ
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
annotationsͺ *7’4
2/+???????????????????????????
2
.__inference_up_sampling2d_layer_call_fn_101801ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
±2?
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_101795ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
3__inference_separable_conv2d_3_layer_call_fn_101830Χ
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
annotationsͺ *7’4
2/+???????????????????????????1
­2ͺ
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_101818Χ
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
annotationsͺ *7’4
2/+???????????????????????????1
2
0__inference_up_sampling2d_1_layer_call_fn_101849ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_101843ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
3__inference_separable_conv2d_4_layer_call_fn_101878Χ
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
annotationsͺ *7’4
2/+???????????????????????????1
­2ͺ
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_101866Χ
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
annotationsͺ *7’4
2/+???????????????????????????1
Σ2Π
)__inference_conv2d_2_layer_call_fn_103560’
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
annotationsͺ *
 
ξ2λ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_103571’
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
annotationsͺ *
 Α
!__inference__wrapped_model_101276'()*+,-./0123456789:;<=>?7’4
-’*
(%
enc_in?????????
ͺ "EͺB
@
mnist_dec_var/,
mnist_dec_var?????????΄
D__inference_conv2d_1_layer_call_and_return_conditional_losses_103470l/07’4
-’*
(%
inputs?????????1
ͺ "-’*
# 
0?????????1
 
)__inference_conv2d_1_layer_call_fn_103459_/07’4
-’*
(%
inputs?????????1
ͺ " ?????????1Ω
D__inference_conv2d_2_layer_call_and_return_conditional_losses_103571>?I’F
?’<
:7
inputs+???????????????????????????1
ͺ "?’<
52
0+???????????????????????????
 ±
)__inference_conv2d_2_layer_call_fn_103560>?I’F
?’<
:7
inputs+???????????????????????????1
ͺ "2/+???????????????????????????²
B__inference_conv2d_layer_call_and_return_conditional_losses_103450l'(7’4
-’*
(%
inputs?????????
ͺ "-’*
# 
0?????????
 
'__inference_conv2d_layer_call_fn_103439_'(7’4
-’*
(%
inputs?????????
ͺ " ?????????£
C__inference_dense_1_layer_call_and_return_conditional_losses_103510\34/’,
%’"
 
inputs?????????1
ͺ "%’"

0?????????1
 {
(__inference_dense_1_layer_call_fn_103499O34/’,
%’"
 
inputs?????????1
ͺ "?????????1‘
A__inference_dense_layer_call_and_return_conditional_losses_103490\12/’,
%’"
 
inputs?????????1
ͺ "%’"

0?????????1
 y
&__inference_dense_layer_call_fn_103479O12/’,
%’"
 
inputs?????????1
ͺ "?????????1έ
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_101341R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ".’+
$!
0??????????????????
 ΄
9__inference_global_average_pooling2d_layer_call_fn_101347wR’O
H’E
C@
inputs4????????????????????????????????????
ͺ "!??????????????????κ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102553'()*+,-./0123456789:;<=>??’<
5’2
(%
enc_in?????????
p 

 
ͺ "?’<
52
0+???????????????????????????
 κ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102609'()*+,-./0123456789:;<=>??’<
5’2
(%
enc_in?????????
p

 
ͺ "?’<
52
0+???????????????????????????
 Ψ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_102912'()*+,-./0123456789:;<=>??’<
5’2
(%
inputs?????????
p 

 
ͺ "-’*
# 
0?????????
 Ψ
H__inference_mnist_ae_var_layer_call_and_return_conditional_losses_103042'()*+,-./0123456789:;<=>??’<
5’2
(%
inputs?????????
p

 
ͺ "-’*
# 
0?????????
 Β
-__inference_mnist_ae_var_layer_call_fn_102276'()*+,-./0123456789:;<=>??’<
5’2
(%
enc_in?????????
p 

 
ͺ "2/+???????????????????????????Β
-__inference_mnist_ae_var_layer_call_fn_102497'()*+,-./0123456789:;<=>??’<
5’2
(%
enc_in?????????
p

 
ͺ "2/+???????????????????????????Β
-__inference_mnist_ae_var_layer_call_fn_102727'()*+,-./0123456789:;<=>??’<
5’2
(%
inputs?????????
p 

 
ͺ "2/+???????????????????????????Β
-__inference_mnist_ae_var_layer_call_fn_102782'()*+,-./0123456789:;<=>??’<
5’2
(%
inputs?????????
p

 
ͺ "2/+???????????????????????????Υ
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_10213056789:;<=>?7’4
-’*
 
dec_in?????????1
p 

 
ͺ "?’<
52
0+???????????????????????????
 Υ
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_10216356789:;<=>?7’4
-’*
 
dec_in?????????1
p

 
ͺ "?’<
52
0+???????????????????????????
 Β
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_103365u56789:;<=>?7’4
-’*
 
inputs?????????1
p 

 
ͺ "-’*
# 
0?????????
 Β
I__inference_mnist_dec_var_layer_call_and_return_conditional_losses_103430u56789:;<=>?7’4
-’*
 
inputs?????????1
p

 
ͺ "-’*
# 
0?????????
 ¬
.__inference_mnist_dec_var_layer_call_fn_101967z56789:;<=>?7’4
-’*
 
dec_in?????????1
p 

 
ͺ "2/+???????????????????????????¬
.__inference_mnist_dec_var_layer_call_fn_102097z56789:;<=>?7’4
-’*
 
dec_in?????????1
p

 
ͺ "2/+???????????????????????????¬
.__inference_mnist_dec_var_layer_call_fn_103273z56789:;<=>?7’4
-’*
 
inputs?????????1
p 

 
ͺ "2/+???????????????????????????¬
.__inference_mnist_dec_var_layer_call_fn_103300z56789:;<=>?7’4
-’*
 
inputs?????????1
p

 
ͺ "2/+???????????????????????????Ε
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101713x'()*+,-./01234?’<
5’2
(%
enc_in?????????
p 

 
ͺ "%’"

0?????????1
 Ε
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_101753x'()*+,-./01234?’<
5’2
(%
enc_in?????????
p

 
ͺ "%’"

0?????????1
 Ε
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_103177x'()*+,-./01234?’<
5’2
(%
inputs?????????
p 

 
ͺ "%’"

0?????????1
 Ε
I__inference_mnist_enc_var_layer_call_and_return_conditional_losses_103246x'()*+,-./01234?’<
5’2
(%
inputs?????????
p

 
ͺ "%’"

0?????????1
 
.__inference_mnist_enc_var_layer_call_fn_101487k'()*+,-./01234?’<
5’2
(%
enc_in?????????
p 

 
ͺ "?????????1
.__inference_mnist_enc_var_layer_call_fn_101673k'()*+,-./01234?’<
5’2
(%
enc_in?????????
p

 
ͺ "?????????1
.__inference_mnist_enc_var_layer_call_fn_103075k'()*+,-./01234?’<
5’2
(%
inputs?????????
p 

 
ͺ "?????????1
.__inference_mnist_enc_var_layer_call_fn_103108k'()*+,-./01234?’<
5’2
(%
inputs?????????
p

 
ͺ "?????????1§
C__inference_reshape_layer_call_and_return_conditional_losses_103551`/’,
%’"
 
inputs?????????1
ͺ "-’*
# 
0?????????
 
(__inference_reshape_layer_call_fn_103537S/’,
%’"
 
inputs?????????1
ͺ " ?????????Μ
D__inference_sampling_layer_call_and_return_conditional_losses_103532Z’W
P’M
KH
"
inputs/0?????????1
"
inputs/1?????????1
ͺ "%’"

0?????????1
 £
)__inference_sampling_layer_call_fn_103516vZ’W
P’M
KH
"
inputs/0?????????1
"
inputs/1?????????1
ͺ "?????????1δ
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_101322,-.I’F
?’<
:7
inputs+???????????????????????????1
ͺ "?’<
52
0+???????????????????????????1
 Ό
3__inference_separable_conv2d_1_layer_call_fn_101334,-.I’F
?’<
:7
inputs+???????????????????????????1
ͺ "2/+???????????????????????????1δ
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_101770567I’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????1
 Ό
3__inference_separable_conv2d_2_layer_call_fn_101782567I’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????1δ
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_10181889:I’F
?’<
:7
inputs+???????????????????????????1
ͺ "?’<
52
0+???????????????????????????1
 Ό
3__inference_separable_conv2d_3_layer_call_fn_10183089:I’F
?’<
:7
inputs+???????????????????????????1
ͺ "2/+???????????????????????????1δ
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_101866;<=I’F
?’<
:7
inputs+???????????????????????????1
ͺ "?’<
52
0+???????????????????????????1
 Ό
3__inference_separable_conv2d_4_layer_call_fn_101878;<=I’F
?’<
:7
inputs+???????????????????????????1
ͺ "2/+???????????????????????????1β
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_101293)*+I’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????1
 Ί
1__inference_separable_conv2d_layer_call_fn_101305)*+I’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????1Ξ
$__inference_signature_wrapper_102672₯'()*+,-./0123456789:;<=>?A’>
’ 
7ͺ4
2
enc_in(%
enc_in?????????"EͺB
@
mnist_dec_var/,
mnist_dec_var?????????ξ
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_101843R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_up_sampling2d_1_layer_call_fn_101849R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????μ
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_101795R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Δ
.__inference_up_sampling2d_layer_call_fn_101801R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????
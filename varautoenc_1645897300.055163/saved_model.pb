¤Þ
ù
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
 "serve*2.6.02unknown8÷
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
Í
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueüBø Bð
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
ã
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
trainable_variables
regularization_losses
	variables
	keras_api
¢
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
trainable_variables
regularization_losses
 	variables
!	keras_api
´
"iter

#beta_1

$beta_2
	%decay
&learning_rate'mÛ(mÜ)mÝ*mÞ+mß,mà-má.mâ/mã0mä1må2mæ3mç4mè5mé6mê7më8mì9mí:mî;mï<mð=mñ>mò?mó'vô(võ)vö*v÷+vø,vù-vú.vû/vü0vý1vþ2vÿ3v4v5v6v7v8v9v:v;v<v=v>v?v
¾
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
¾
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
­
@non_trainable_variables
trainable_variables
Alayer_regularization_losses
regularization_losses
	variables
Bmetrics

Clayers
Dlayer_metrics
 
h

'kernel
(bias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api

)depthwise_kernel
*pointwise_kernel
+bias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api

,depthwise_kernel
-pointwise_kernel
.bias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
h

/kernel
0bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
R
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

1kernel
2bias
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
h

3kernel
4bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
R
atrainable_variables
b	variables
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
 
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
­
enon_trainable_variables
trainable_variables
flayer_regularization_losses
regularization_losses
	variables
gmetrics

hlayers
ilayer_metrics
 
R
jtrainable_variables
k	variables
lregularization_losses
m	keras_api

5depthwise_kernel
6pointwise_kernel
7bias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
R
rtrainable_variables
s	variables
tregularization_losses
u	keras_api

8depthwise_kernel
9pointwise_kernel
:bias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api

;depthwise_kernel
<pointwise_kernel
=bias
~trainable_variables
	variables
regularization_losses
	keras_api
l

>kernel
?bias
trainable_variables
	variables
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
 
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
²
non_trainable_variables
trainable_variables
 layer_regularization_losses
regularization_losses
 	variables
metrics
layers
layer_metrics
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
SQ
VARIABLE_VALUEdense/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUE
dense/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_1/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_1/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEseparable_conv2d_2/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEseparable_conv2d_3/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEseparable_conv2d_4/bias1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_2/kernel1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_2/bias1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
1
2
 

'0
(1

'0
(1
 
²
non_trainable_variables
Etrainable_variables
 layer_regularization_losses
F	variables
Gregularization_losses
metrics
layers
layer_metrics
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
non_trainable_variables
Itrainable_variables
 layer_regularization_losses
J	variables
Kregularization_losses
metrics
layers
layer_metrics
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
non_trainable_variables
Mtrainable_variables
 layer_regularization_losses
N	variables
Oregularization_losses
metrics
layers
layer_metrics

/0
01

/0
01
 
²
non_trainable_variables
Qtrainable_variables
 layer_regularization_losses
R	variables
Sregularization_losses
metrics
layers
layer_metrics
 
 
 
²
 non_trainable_variables
Utrainable_variables
 ¡layer_regularization_losses
V	variables
Wregularization_losses
¢metrics
£layers
¤layer_metrics

10
21

10
21
 
²
¥non_trainable_variables
Ytrainable_variables
 ¦layer_regularization_losses
Z	variables
[regularization_losses
§metrics
¨layers
©layer_metrics

30
41

30
41
 
²
ªnon_trainable_variables
]trainable_variables
 «layer_regularization_losses
^	variables
_regularization_losses
¬metrics
­layers
®layer_metrics
 
 
 
²
¯non_trainable_variables
atrainable_variables
 °layer_regularization_losses
b	variables
cregularization_losses
±metrics
²layers
³layer_metrics
 
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
²
´non_trainable_variables
jtrainable_variables
 µlayer_regularization_losses
k	variables
lregularization_losses
¶metrics
·layers
¸layer_metrics
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
¹non_trainable_variables
ntrainable_variables
 ºlayer_regularization_losses
o	variables
pregularization_losses
»metrics
¼layers
½layer_metrics
 
 
 
²
¾non_trainable_variables
rtrainable_variables
 ¿layer_regularization_losses
s	variables
tregularization_losses
Àmetrics
Álayers
Âlayer_metrics
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
Ãnon_trainable_variables
vtrainable_variables
 Älayer_regularization_losses
w	variables
xregularization_losses
Åmetrics
Ælayers
Çlayer_metrics
 
 
 
²
Ènon_trainable_variables
ztrainable_variables
 Élayer_regularization_losses
{	variables
|regularization_losses
Êmetrics
Ëlayers
Ìlayer_metrics
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
Ínon_trainable_variables
~trainable_variables
 Îlayer_regularization_losses
	variables
regularization_losses
Ïmetrics
Ðlayers
Ñlayer_metrics

>0
?1

>0
?1
 
µ
Ònon_trainable_variables
trainable_variables
 Ólayer_regularization_losses
	variables
regularization_losses
Ômetrics
Õlayers
Ölayer_metrics
 
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
8

×total

Øcount
Ù	variables
Ú	keras_api
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
×0
Ø1

Ù	variables
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
vt
VARIABLE_VALUEAdam/dense/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/dense/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_2/bias/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_3/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_4/bias/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_2/kernel/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_2/bias/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
vt
VARIABLE_VALUEAdam/dense/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/dense/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_2/bias/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_3/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_4/bias/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_2/kernel/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_2/bias/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_enc_inPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
ê
StatefulPartitionedCallStatefulPartitionedCallserving_default_enc_inconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasconv2d_2/kernelconv2d_2/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_97076
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÿ!
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_98244

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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_98500¿ñ
«	
Ì
2__inference_separable_conv2d_2_layer_call_fn_96186

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
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_961742
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
þ

ñ
@__inference_dense_layer_call_and_return_conditional_losses_97894

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¯
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_96199

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
±
f
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_96247

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
é
ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97874

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
¤

K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95697

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
é
Ë
-__inference_mnist_dec_var_layer_call_fn_97704

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
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_964492
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
¦

M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_96222

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
þ

ñ
@__inference_dense_layer_call_and_return_conditional_losses_95818

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¦

M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95726

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


&__inference_conv2d_layer_call_fn_97843

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
A__inference_conv2d_layer_call_and_return_conditional_losses_957692
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
Ö
ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_96339

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

^
B__inference_reshape_layer_call_and_return_conditional_losses_97955

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
¥-

H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96013

inputs&
conv2d_95976:
conv2d_95978:0
separable_conv2d_95981:0
separable_conv2d_95983:1$
separable_conv2d_95985:12
separable_conv2d_1_95988:12
separable_conv2d_1_95990:11&
separable_conv2d_1_95992:1(
conv2d_1_95995:11
conv2d_1_95997:1
dense_96001:11
dense_96003:1
dense_1_96006:11
dense_1_96008:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ sampling/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_95976conv2d_95978*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_957692 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_95981separable_conv2d_95983separable_conv2d_95985*
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
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_956972*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95988separable_conv2d_1_95990separable_conv2d_1_95992*
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
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_957262,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_95995conv2d_1_95997*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_958002"
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
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_957452*
(global_average_pooling2d/PartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_96001dense_96003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_958182
dense/StatefulPartitionedCallº
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_96006dense_1_96008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_958352!
dense_1/StatefulPartitionedCall¹
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_958572"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

IdentityÏ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§	
Ê
0__inference_separable_conv2d_layer_call_fn_95709

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
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_956972
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
ì

%__inference_dense_layer_call_fn_97883

inputs
unknown:11
	unknown_0:1
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_958182
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
»'

H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_96449

inputs2
separable_conv2d_2_96420:2
separable_conv2d_2_96422:1&
separable_conv2d_2_96424:12
separable_conv2d_3_96428:12
separable_conv2d_3_96430:11&
separable_conv2d_3_96432:12
separable_conv2d_4_96436:12
separable_conv2d_4_96438:11&
separable_conv2d_4_96440:1(
conv2d_2_96443:1
conv2d_2_96445:
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
B__inference_reshape_layer_call_and_return_conditional_losses_963032
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96420separable_conv2d_2_96422separable_conv2d_2_96424*
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
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_961742,
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
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_961992
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96428separable_conv2d_3_96430separable_conv2d_3_96432*
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
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_962222,
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
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_962472!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96436separable_conv2d_4_96438separable_conv2d_4_96440*
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
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_962702,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96443conv2d_2_96445*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_963392"
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
¥-

H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96157

enc_in&
conv2d_96120:
conv2d_96122:0
separable_conv2d_96125:0
separable_conv2d_96127:1$
separable_conv2d_96129:12
separable_conv2d_1_96132:12
separable_conv2d_1_96134:11&
separable_conv2d_1_96136:1(
conv2d_1_96139:11
conv2d_1_96141:1
dense_96145:11
dense_96147:1
dense_1_96150:11
dense_1_96152:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ sampling/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_96120conv2d_96122*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_957692 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_96125separable_conv2d_96127separable_conv2d_96129*
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
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_956972*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_96132separable_conv2d_1_96134separable_conv2d_1_96136*
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
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_957262,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_96139conv2d_1_96141*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_958002"
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
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_957452*
(global_average_pooling2d/PartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_96145dense_96147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_958182
dense/StatefulPartitionedCallº
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_96150dense_1_96152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_958352!
dense_1/StatefulPartitionedCall¹
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_958572"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

IdentityÏ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
¥-

H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96117

enc_in&
conv2d_96080:
conv2d_96082:0
separable_conv2d_96085:0
separable_conv2d_96087:1$
separable_conv2d_96089:12
separable_conv2d_1_96092:12
separable_conv2d_1_96094:11&
separable_conv2d_1_96096:1(
conv2d_1_96099:11
conv2d_1_96101:1
dense_96105:11
dense_96107:1
dense_1_96110:11
dense_1_96112:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ sampling/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallenc_inconv2d_96080conv2d_96082*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_957692 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_96085separable_conv2d_96087separable_conv2d_96089*
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
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_956972*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_96092separable_conv2d_1_96094separable_conv2d_1_96096*
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
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_957262,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_96099conv2d_1_96101*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_958002"
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
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_957452*
(global_average_pooling2d/PartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_96105dense_96107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_958182
dense/StatefulPartitionedCallº
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_96110dense_1_96112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_958352!
dense_1/StatefulPartitionedCall¹
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_958572"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

IdentityÏ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
¦

M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_96174

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
»e
¿
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_97581

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
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢'separable_conv2d/BiasAdd/ReadVariableOp¢0separable_conv2d/separable_conv2d/ReadVariableOp¢2separable_conv2d/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_1/BiasAdd/ReadVariableOp¢2separable_conv2d_1/separable_conv2d/ReadVariableOp¢4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ª
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
global_average_pooling2d/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense/MatMul/ReadVariableOp¥
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense/BiasAddj

dense/SeluSeludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

dense/Selu¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense_1/MatMul/ReadVariableOp«
dense_1/MatMulMatMul&global_average_pooling2d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense_1/BiasAddp
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
sampling/random_normal/stddevé
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed2³Ä½2-
+sampling/random_normal/RandomStandardNormalÏ
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/random_normal/mul±
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
:ÿÿÿÿÿÿÿÿÿ12
sampling/truedivk
sampling/ExpExpsampling/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/Exp
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/mul
sampling/addAddV2sampling/mul:z:0dense/Selu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/addk
IdentityIdentitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityø
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
ý
#__inference_signature_wrapper_97076

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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_956802
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
«	
Ì
2__inference_separable_conv2d_3_layer_call_fn_96234

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
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_962222
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
Ë
-__inference_mnist_dec_var_layer_call_fn_97677

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
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_963462
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
»'

H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_96346

inputs2
separable_conv2d_2_96305:2
separable_conv2d_2_96307:1&
separable_conv2d_2_96309:12
separable_conv2d_3_96313:12
separable_conv2d_3_96315:11&
separable_conv2d_3_96317:12
separable_conv2d_4_96321:12
separable_conv2d_4_96323:11&
separable_conv2d_4_96325:1(
conv2d_2_96340:1
conv2d_2_96342:
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
B__inference_reshape_layer_call_and_return_conditional_losses_963032
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96305separable_conv2d_2_96307separable_conv2d_2_96309*
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
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_961742,
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
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_961992
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96313separable_conv2d_3_96315separable_conv2d_3_96317*
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
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_962222,
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
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_962472!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96321separable_conv2d_4_96323separable_conv2d_4_96325*
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
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_962702,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96340conv2d_2_96342*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_963392"
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


,__inference_mnist_ae_var_layer_call_fn_97131

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
identity¢StatefulPartitionedCallÍ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_966272
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·û
Ö
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97316

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
identity¢-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp¢,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp¢7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp¢@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp¢Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp¢@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp¢Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp¢@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp¢Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1¢+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp¢*mnist_enc_var/conv2d/Conv2D/ReadVariableOp¢-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp¢,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp¢*mnist_enc_var/dense/BiasAdd/ReadVariableOp¢)mnist_enc_var/dense/MatMul/ReadVariableOp¢,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp¢+mnist_enc_var/dense_1/MatMul/ReadVariableOp¢5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp¢>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp¢@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1¢7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp¢@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp¢Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ô
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpReadVariableOp3mnist_enc_var_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpâ
mnist_enc_var/conv2d/Conv2DConv2Dinputs2mnist_enc_var/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_enc_var/conv2d/Conv2DË
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpReadVariableOp4mnist_enc_var_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpÜ
mnist_enc_var/conv2d/BiasAddBiasAdd$mnist_enc_var/conv2d/Conv2D:output:03mnist_enc_var/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_enc_var/conv2d/BiasAdd
mnist_enc_var/conv2d/SeluSelu%mnist_enc_var/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1Ç
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            27
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeÏ
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateâ
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative'mnist_enc_var/conv2d/Selu:activations:0Fmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2;
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseÝ
/mnist_enc_var/separable_conv2d/separable_conv2dConv2DBmnist_enc_var/separable_conv2d/separable_conv2d/depthwise:output:0Hmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
21
/mnist_enc_var/separable_conv2d/separable_conv2dé
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp>mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype027
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp
&mnist_enc_var/separable_conv2d/BiasAddBiasAdd8mnist_enc_var/separable_conv2d/separable_conv2d:output:0=mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12(
&mnist_enc_var/separable_conv2d/BiasAdd½
#mnist_enc_var/separable_conv2d/SeluSelu/mnist_enc_var/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12%
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
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ë
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeÓ
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateò
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative1mnist_enc_var/separable_conv2d/Selu:activations:0Hmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2=
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseå
1mnist_enc_var/separable_conv2d_1/separable_conv2dConv2DDmnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise:output:0Jmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_enc_var/separable_conv2d_1/separable_conv2dï
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp
(mnist_enc_var/separable_conv2d_1/BiasAddBiasAdd:mnist_enc_var/separable_conv2d_1/separable_conv2d:output:0?mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_enc_var/separable_conv2d_1/BiasAddÃ
%mnist_enc_var/separable_conv2d_1/SeluSelu1mnist_enc_var/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
%mnist_enc_var/separable_conv2d_1/SeluÚ
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5mnist_enc_var_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02.
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp
mnist_enc_var/conv2d_1/Conv2DConv2D3mnist_enc_var/separable_conv2d_1/Selu:activations:04mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
mnist_enc_var/conv2d_1/Conv2DÑ
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6mnist_enc_var_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02/
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpä
mnist_enc_var/conv2d_1/BiasAddBiasAdd&mnist_enc_var/conv2d_1/Conv2D:output:05mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12 
mnist_enc_var/conv2d_1/BiasAdd¥
mnist_enc_var/conv2d_1/SeluSelu'mnist_enc_var/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/conv2d_1/SeluÏ
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indices
+mnist_enc_var/global_average_pooling2d/MeanMean)mnist_enc_var/conv2d_1/Selu:activations:0Fmnist_enc_var/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12-
+mnist_enc_var/global_average_pooling2d/MeanÉ
)mnist_enc_var/dense/MatMul/ReadVariableOpReadVariableOp2mnist_enc_var_dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02+
)mnist_enc_var/dense/MatMul/ReadVariableOpÝ
mnist_enc_var/dense/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:01mnist_enc_var/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense/MatMulÈ
*mnist_enc_var/dense/BiasAdd/ReadVariableOpReadVariableOp3mnist_enc_var_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02,
*mnist_enc_var/dense/BiasAdd/ReadVariableOpÑ
mnist_enc_var/dense/BiasAddBiasAdd$mnist_enc_var/dense/MatMul:product:02mnist_enc_var/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense/BiasAdd
mnist_enc_var/dense/SeluSelu$mnist_enc_var/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense/SeluÏ
+mnist_enc_var/dense_1/MatMul/ReadVariableOpReadVariableOp4mnist_enc_var_dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02-
+mnist_enc_var/dense_1/MatMul/ReadVariableOpã
mnist_enc_var/dense_1/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:03mnist_enc_var/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense_1/MatMulÎ
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpReadVariableOp5mnist_enc_var_dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02.
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpÙ
mnist_enc_var/dense_1/BiasAddBiasAdd&mnist_enc_var/dense_1/MatMul:product:04mnist_enc_var/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense_1/BiasAdd
mnist_enc_var/dense_1/SeluSelu&mnist_enc_var/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed22;
9mnist_enc_var/sampling/random_normal/RandomStandardNormal
(mnist_enc_var/sampling/random_normal/mulMulBmnist_enc_var/sampling/random_normal/RandomStandardNormal:output:04mnist_enc_var/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_enc_var/sampling/random_normal/mulé
$mnist_enc_var/sampling/random_normalAddV2,mnist_enc_var/sampling/random_normal/mul:z:02mnist_enc_var/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_enc_var/sampling/random_normal
 mnist_enc_var/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 mnist_enc_var/sampling/truediv/yÒ
mnist_enc_var/sampling/truedivRealDiv(mnist_enc_var/dense_1/Selu:activations:0)mnist_enc_var/sampling/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12 
mnist_enc_var/sampling/truediv
mnist_enc_var/sampling/ExpExp"mnist_enc_var/sampling/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/sampling/Exp»
mnist_enc_var/sampling/mulMul(mnist_enc_var/sampling/random_normal:z:0mnist_enc_var/sampling/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/sampling/mul»
mnist_enc_var/sampling/addAddV2mnist_enc_var/sampling/mul:z:0&mnist_enc_var/dense/Selu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
)mnist_dec_var/reshape/strided_slice/stack¤
+mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_1¤
+mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_2æ
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
%mnist_dec_var/reshape/Reshape/shape/3¾
#mnist_dec_var/reshape/Reshape/shapePack,mnist_dec_var/reshape/strided_slice:output:0.mnist_dec_var/reshape/Reshape/shape/1:output:0.mnist_dec_var/reshape/Reshape/shape/2:output:0.mnist_dec_var/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#mnist_dec_var/reshape/Reshape/shapeÑ
mnist_dec_var/reshape/ReshapeReshapemnist_enc_var/sampling/add:z:0,mnist_dec_var/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ë
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeÓ
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateç
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative&mnist_dec_var/reshape/Reshape:output:0Hmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseå
1mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_2/separable_conv2dï
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_2/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_dec_var/separable_conv2d_2/BiasAddÃ
%mnist_dec_var/separable_conv2d_2/SeluSelu1mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
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
#mnist_dec_var/up_sampling2d/Const_1È
mnist_dec_var/up_sampling2d/mulMul*mnist_dec_var/up_sampling2d/Const:output:0,mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2!
mnist_dec_var/up_sampling2d/mul»
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_2/Selu:activations:0#mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
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
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ë
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeÓ
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeImnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseå
1mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_3/separable_conv2dï
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_3/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_dec_var/separable_conv2d_3/BiasAddÃ
%mnist_dec_var/separable_conv2d_3/SeluSelu1mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
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
%mnist_dec_var/up_sampling2d_1/Const_1Ð
!mnist_dec_var/up_sampling2d_1/mulMul,mnist_dec_var/up_sampling2d_1/Const:output:0.mnist_dec_var/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2#
!mnist_dec_var/up_sampling2d_1/mulÁ
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_3/Selu:activations:0%mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
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
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ë
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeÓ
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeKmnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseå
1mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_4/separable_conv2dï
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_4/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_dec_var/separable_conv2d_4/BiasAddÃ
%mnist_dec_var/separable_conv2d_4/SeluSelu1mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
%mnist_dec_var/separable_conv2d_4/SeluÚ
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02.
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp
mnist_dec_var/conv2d_2/Conv2DConv2D3mnist_dec_var/separable_conv2d_4/Selu:activations:04mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_dec_var/conv2d_2/Conv2DÑ
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpä
mnist_dec_var/conv2d_2/BiasAddBiasAdd&mnist_dec_var/conv2d_2/Conv2D:output:05mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
mnist_dec_var/conv2d_2/BiasAdd¥
mnist_dec_var/conv2d_2/SeluSelu'mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec_var/conv2d_2/Selu
IdentityIdentity)mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityá
NoOpNoOp.^mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-^mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp8^mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1,^mnist_enc_var/conv2d/BiasAdd/ReadVariableOp+^mnist_enc_var/conv2d/Conv2D/ReadVariableOp.^mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp-^mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp+^mnist_enc_var/dense/BiasAdd/ReadVariableOp*^mnist_enc_var/dense/MatMul/ReadVariableOp-^mnist_enc_var/dense_1/BiasAdd/ReadVariableOp,^mnist_enc_var/dense_1/MatMul/ReadVariableOp6^mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp?^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpA^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_18^mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpA^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpC^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2^
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_mnist_ae_var_layer_call_fn_96680

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
identity¢StatefulPartitionedCallÍ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_966272
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
¾

-__inference_mnist_enc_var_layer_call_fn_95891

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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_958602
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
é
Ë
-__inference_mnist_dec_var_layer_call_fn_96501

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
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_964492
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
¤
q
(__inference_sampling_layer_call_fn_97920
inputs_0
inputs_1
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
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
GPU2*0J 8 *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_958572
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
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ1:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
"
_user_specified_name
inputs/1
é
Ë
-__inference_mnist_dec_var_layer_call_fn_96371

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
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_963462
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
Ö
ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97975

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
¸
Å	
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_96627

inputs-
mnist_enc_var_96574:!
mnist_enc_var_96576:-
mnist_enc_var_96578:-
mnist_enc_var_96580:1!
mnist_enc_var_96582:1-
mnist_enc_var_96584:1-
mnist_enc_var_96586:11!
mnist_enc_var_96588:1-
mnist_enc_var_96590:11!
mnist_enc_var_96592:1%
mnist_enc_var_96594:11!
mnist_enc_var_96596:1%
mnist_enc_var_96598:11!
mnist_enc_var_96600:1-
mnist_dec_var_96603:-
mnist_dec_var_96605:1!
mnist_dec_var_96607:1-
mnist_dec_var_96609:1-
mnist_dec_var_96611:11!
mnist_dec_var_96613:1-
mnist_dec_var_96615:1-
mnist_dec_var_96617:11!
mnist_dec_var_96619:1-
mnist_dec_var_96621:1!
mnist_dec_var_96623:
identity¢%mnist_dec_var/StatefulPartitionedCall¢%mnist_enc_var/StatefulPartitionedCallÁ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallinputsmnist_enc_var_96574mnist_enc_var_96576mnist_enc_var_96578mnist_enc_var_96580mnist_enc_var_96582mnist_enc_var_96584mnist_enc_var_96586mnist_enc_var_96588mnist_enc_var_96590mnist_enc_var_96592mnist_enc_var_96594mnist_enc_var_96596mnist_enc_var_96598mnist_enc_var_96600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_958602'
%mnist_enc_var/StatefulPartitionedCall¾
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_96603mnist_dec_var_96605mnist_dec_var_96607mnist_dec_var_96609mnist_dec_var_96611mnist_dec_var_96613mnist_dec_var_96615mnist_dec_var_96617mnist_dec_var_96619mnist_dec_var_96621mnist_dec_var_96623*
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_963462'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
K
/__inference_up_sampling2d_1_layer_call_fn_96253

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
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_962472
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
ç
ú
A__inference_conv2d_layer_call_and_return_conditional_losses_95769

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
ð

'__inference_dense_1_layer_call_fn_97903

inputs
unknown:11
	unknown_0:1
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_958352
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¦

M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_96270

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
º¨
¹!
 __inference__wrapped_model_95680

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
identity¢:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp¢9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp¢Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp¢Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp¢Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp¢Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp¢Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp¢Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp¢Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1¢8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp¢7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp¢:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp¢9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp¢7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp¢6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp¢9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp¢8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp¢Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp¢Kmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp¢Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1¢Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp¢Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp¢Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1û
7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOpReadVariableOp@mnist_ae_var_mnist_enc_var_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp
(mnist_ae_var/mnist_enc_var/conv2d/Conv2DConv2Denc_in?mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2*
(mnist_ae_var/mnist_enc_var/conv2d/Conv2Dò
8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOpReadVariableOpAmnist_ae_var_mnist_enc_var_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp
)mnist_ae_var/mnist_enc_var/conv2d/BiasAddBiasAdd1mnist_ae_var/mnist_enc_var/conv2d/Conv2D:output:0@mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)mnist_ae_var/mnist_enc_var/conv2d/BiasAddÆ
&mnist_ae_var/mnist_enc_var/conv2d/SeluSelu2mnist_ae_var/mnist_enc_var/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
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
Mmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1á
Bmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2D
Bmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/Shapeé
Jmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2L
Jmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rate
Fmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative4mnist_ae_var/mnist_enc_var/conv2d/Selu:activations:0Smnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2H
Fmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/depthwise
<mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2dConv2DOmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/depthwise:output:0Umnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2>
<mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d
Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOpKmnist_ae_var_mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02D
Bmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpÂ
3mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAddBiasAddEmnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d:output:0Jmnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ125
3mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAddä
0mnist_ae_var/mnist_enc_var/separable_conv2d/SeluSelu<mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ122
0mnist_ae_var/mnist_enc_var/separable_conv2d/Selu½
Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpÃ
Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_enc_var_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02Q
Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1å
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2F
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/Shapeí
Lmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rate¦
Hmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative>mnist_ae_var/mnist_enc_var/separable_conv2d/Selu:activations:0Umnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise
>mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2dConv2DQmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpÊ
5mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAddBiasAddGmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d:output:0Lmnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ127
5mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAddê
2mnist_ae_var/mnist_enc_var/separable_conv2d_1/SeluSelu>mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ124
2mnist_ae_var/mnist_enc_var/separable_conv2d_1/Selu
9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpReadVariableOpBmnist_ae_var_mnist_enc_var_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02;
9mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpÉ
*mnist_ae_var/mnist_enc_var/conv2d_1/Conv2DConv2D@mnist_ae_var/mnist_enc_var/separable_conv2d_1/Selu:activations:0Amnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2,
*mnist_ae_var/mnist_enc_var/conv2d_1/Conv2Dø
:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpCmnist_ae_var_mnist_enc_var_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02<
:mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp
+mnist_ae_var/mnist_enc_var/conv2d_1/BiasAddBiasAdd3mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D:output:0Bmnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12-
+mnist_ae_var/mnist_enc_var/conv2d_1/BiasAddÌ
(mnist_ae_var/mnist_enc_var/conv2d_1/SeluSelu4mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_ae_var/mnist_enc_var/conv2d_1/Selué
Jmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2L
Jmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean/reduction_indices»
8mnist_ae_var/mnist_enc_var/global_average_pooling2d/MeanMean6mnist_ae_var/mnist_enc_var/conv2d_1/Selu:activations:0Smnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12:
8mnist_ae_var/mnist_enc_var/global_average_pooling2d/Meanð
6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOpReadVariableOp?mnist_ae_var_mnist_enc_var_dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype028
6mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp
'mnist_ae_var/mnist_enc_var/dense/MatMulMatMulAmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean:output:0>mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_ae_var/mnist_enc_var/dense/MatMulï
7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOpReadVariableOp@mnist_ae_var_mnist_enc_var_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp
(mnist_ae_var/mnist_enc_var/dense/BiasAddBiasAdd1mnist_ae_var/mnist_enc_var/dense/MatMul:product:0?mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_ae_var/mnist_enc_var/dense/BiasAdd»
%mnist_ae_var/mnist_enc_var/dense/SeluSelu1mnist_ae_var/mnist_enc_var/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
%mnist_ae_var/mnist_enc_var/dense/Seluö
8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOpReadVariableOpAmnist_ae_var_mnist_enc_var_dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02:
8mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp
)mnist_ae_var/mnist_enc_var/dense_1/MatMulMatMulAmnist_ae_var/mnist_enc_var/global_average_pooling2d/Mean:output:0@mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12+
)mnist_ae_var/mnist_enc_var/dense_1/MatMulõ
9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOpReadVariableOpBmnist_ae_var_mnist_enc_var_dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02;
9mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp
*mnist_ae_var/mnist_enc_var/dense_1/BiasAddBiasAdd3mnist_ae_var/mnist_enc_var/dense_1/MatMul:product:0Amnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12,
*mnist_ae_var/mnist_enc_var/dense_1/BiasAddÁ
'mnist_ae_var/mnist_enc_var/dense_1/SeluSelu3mnist_ae_var/mnist_enc_var/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_ae_var/mnist_enc_var/dense_1/Selu»
)mnist_ae_var/mnist_enc_var/sampling/ShapeShape5mnist_ae_var/mnist_enc_var/dense_1/Selu:activations:0*
T0*
_output_shapes
:2+
)mnist_ae_var/mnist_enc_var/sampling/Shapeµ
6mnist_ae_var/mnist_enc_var/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6mnist_ae_var/mnist_enc_var/sampling/random_normal/mean¹
8mnist_ae_var/mnist_enc_var/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8mnist_ae_var/mnist_enc_var/sampling/random_normal/stddev¹
Fmnist_ae_var/mnist_enc_var/sampling/random_normal/RandomStandardNormalRandomStandardNormal2mnist_ae_var/mnist_enc_var/sampling/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed2à³Q2H
Fmnist_ae_var/mnist_enc_var/sampling/random_normal/RandomStandardNormal»
5mnist_ae_var/mnist_enc_var/sampling/random_normal/mulMulOmnist_ae_var/mnist_enc_var/sampling/random_normal/RandomStandardNormal:output:0Amnist_ae_var/mnist_enc_var/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ127
5mnist_ae_var/mnist_enc_var/sampling/random_normal/mul
1mnist_ae_var/mnist_enc_var/sampling/random_normalAddV29mnist_ae_var/mnist_enc_var/sampling/random_normal/mul:z:0?mnist_ae_var/mnist_enc_var/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ123
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
:ÿÿÿÿÿÿÿÿÿ12-
+mnist_ae_var/mnist_enc_var/sampling/truediv¼
'mnist_ae_var/mnist_enc_var/sampling/ExpExp/mnist_ae_var/mnist_enc_var/sampling/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_ae_var/mnist_enc_var/sampling/Expï
'mnist_ae_var/mnist_enc_var/sampling/mulMul5mnist_ae_var/mnist_enc_var/sampling/random_normal:z:0+mnist_ae_var/mnist_enc_var/sampling/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_ae_var/mnist_enc_var/sampling/mulï
'mnist_ae_var/mnist_enc_var/sampling/addAddV2+mnist_ae_var/mnist_enc_var/sampling/mul:z:03mnist_ae_var/mnist_enc_var/dense/Selu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12)
'mnist_ae_var/mnist_enc_var/sampling/add¯
(mnist_ae_var/mnist_dec_var/reshape/ShapeShape+mnist_ae_var/mnist_enc_var/sampling/add:z:0*
T0*
_output_shapes
:2*
(mnist_ae_var/mnist_dec_var/reshape/Shapeº
6mnist_ae_var/mnist_dec_var/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack¾
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_1¾
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_2´
0mnist_ae_var/mnist_dec_var/reshape/strided_sliceStridedSlice1mnist_ae_var/mnist_dec_var/reshape/Shape:output:0?mnist_ae_var/mnist_dec_var/reshape/strided_slice/stack:output:0Amnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_1:output:0Amnist_ae_var/mnist_dec_var/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0mnist_ae_var/mnist_dec_var/reshape/strided_sliceª
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/1ª
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2mnist_ae_var/mnist_dec_var/reshape/Reshape/shape/2ª
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
:ÿÿÿÿÿÿÿÿÿ2,
*mnist_ae_var/mnist_dec_var/reshape/Reshape½
Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02O
Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpÃ
Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_dec_var_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype02Q
Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1å
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/Shapeí
Lmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rate
Hmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative3mnist_ae_var/mnist_dec_var/reshape/Reshape:output:0Umnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise
>mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DQmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpÊ
5mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAddBiasAddGmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0Lmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ127
5mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAddê
2mnist_ae_var/mnist_dec_var/separable_conv2d_2/SeluSelu>mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ124
2mnist_ae_var/mnist_dec_var/separable_conv2d_2/Selu±
.mnist_ae_var/mnist_dec_var/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      20
.mnist_ae_var/mnist_dec_var/up_sampling2d/Constµ
0mnist_ae_var/mnist_dec_var/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      22
0mnist_ae_var/mnist_dec_var/up_sampling2d/Const_1ü
,mnist_ae_var/mnist_dec_var/up_sampling2d/mulMul7mnist_ae_var/mnist_dec_var/up_sampling2d/Const:output:09mnist_ae_var/mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2.
,mnist_ae_var/mnist_dec_var/up_sampling2d/mulï
Emnist_ae_var/mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor@mnist_ae_var/mnist_dec_var/separable_conv2d_2/Selu:activations:00mnist_ae_var/mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2G
Emnist_ae_var/mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor½
Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpÃ
Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_dec_var_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02Q
Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1å
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/Shapeí
Lmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rate¾
Hmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeVmnist_ae_var/mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Umnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise
>mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DQmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpÊ
5mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAddBiasAddGmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0Lmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ127
5mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAddê
2mnist_ae_var/mnist_dec_var/separable_conv2d_3/SeluSelu>mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ124
2mnist_ae_var/mnist_dec_var/separable_conv2d_3/Seluµ
0mnist_ae_var/mnist_dec_var/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      22
0mnist_ae_var/mnist_dec_var/up_sampling2d_1/Const¹
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
.mnist_ae_var/mnist_dec_var/up_sampling2d_1/mulõ
Gmnist_ae_var/mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor@mnist_ae_var/mnist_dec_var/separable_conv2d_3/Selu:activations:02mnist_ae_var/mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
half_pixel_centers(2I
Gmnist_ae_var/mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor½
Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpVmnist_ae_var_mnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02O
Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpÃ
Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpXmnist_ae_var_mnist_dec_var_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:11*
dtype02Q
Omnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1å
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      2F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/Shapeí
Lmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2N
Lmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateÀ
Hmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeXmnist_ae_var/mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Umnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2J
Hmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise
>mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DQmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Wmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
2@
>mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpMmnist_ae_var_mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02F
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpÊ
5mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAddBiasAddGmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0Lmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ127
5mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAddê
2mnist_ae_var/mnist_dec_var/separable_conv2d_4/SeluSelu>mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ124
2mnist_ae_var/mnist_dec_var/separable_conv2d_4/Selu
9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOpBmnist_ae_var_mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02;
9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpÉ
*mnist_ae_var/mnist_dec_var/conv2d_2/Conv2DConv2D@mnist_ae_var/mnist_dec_var/separable_conv2d_4/Selu:activations:0Amnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2,
*mnist_ae_var/mnist_dec_var/conv2d_2/Conv2Dø
:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpCmnist_ae_var_mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp
+mnist_ae_var/mnist_dec_var/conv2d_2/BiasAddBiasAdd3mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D:output:0Bmnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+mnist_ae_var/mnist_dec_var/conv2d_2/BiasAddÌ
(mnist_ae_var/mnist_dec_var/conv2d_2/SeluSelu4mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(mnist_ae_var/mnist_dec_var/conv2d_2/Selu
IdentityIdentity6mnist_ae_var/mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¦
NoOpNoOp;^mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:^mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpE^mnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1E^mnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1E^mnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_19^mnist_ae_var/mnist_enc_var/conv2d/BiasAdd/ReadVariableOp8^mnist_ae_var/mnist_enc_var/conv2d/Conv2D/ReadVariableOp;^mnist_ae_var/mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:^mnist_ae_var/mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp8^mnist_ae_var/mnist_enc_var/dense/BiasAdd/ReadVariableOp7^mnist_ae_var/mnist_enc_var/dense/MatMul/ReadVariableOp:^mnist_ae_var/mnist_enc_var/dense_1/BiasAdd/ReadVariableOp9^mnist_ae_var/mnist_enc_var/dense_1/MatMul/ReadVariableOpC^mnist_ae_var/mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpL^mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpN^mnist_ae_var/mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1E^mnist_ae_var/mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpN^mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpP^mnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2x
:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:mnist_ae_var/mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp2v
9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp9mnist_ae_var/mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp2
Dmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp2¢
Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_12
Dmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp2¢
Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_12
Dmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpDmnist_ae_var/mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp2
Mmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp2¢
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
Mmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpMmnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp2¢
Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Omnist_ae_var/mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
«
'
__inference__traced_save_98244
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
ShardedFilename½,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*Ï+
valueÅ+BÂ+SB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*»
value±B®SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÚ%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop7savev2_adam_separable_conv2d_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_2_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_3_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop7savev2_adam_separable_conv2d_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_2_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_3_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	2
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

identity_1Identity_1:output:0*Õ
_input_shapesÃ
À: : : : : : ::::1:1:1:11:1:11:1:11:1:11:1::1:1:1:11:1:1:11:1:1:: : ::::1:1:1:11:1:11:1:11:1:11:1::1:1:1:11:1:1:11:1:1:::::1:1:1:11:1:11:1:11:1:11:1::1:1:1:11:1:1:11:1:1:: 2(
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
»e
¿
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_97650

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
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢'separable_conv2d/BiasAdd/ReadVariableOp¢0separable_conv2d/separable_conv2d/ReadVariableOp¢2separable_conv2d/separable_conv2d/ReadVariableOp_1¢)separable_conv2d_1/BiasAdd/ReadVariableOp¢2separable_conv2d_1/separable_conv2d/ReadVariableOp¢4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ª
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
global_average_pooling2d/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense/MatMul/ReadVariableOp¥
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense/BiasAddj

dense/SeluSeludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

dense/Selu¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02
dense_1/MatMul/ReadVariableOp«
dense_1/MatMulMatMul&global_average_pooling2d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
dense_1/BiasAddp
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
sampling/random_normal/stddevé
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed2¹³2-
+sampling/random_normal/RandomStandardNormalÏ
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/random_normal/mul±
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
:ÿÿÿÿÿÿÿÿÿ12
sampling/truedivk
sampling/ExpExpsampling/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/Exp
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/mul
sampling/addAddV2sampling/mul:z:0dense/Selu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
sampling/addk
IdentityIdentitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identityø
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_95800

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

ó
B__inference_dense_1_layer_call_and_return_conditional_losses_95835

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ôh
Ã
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_97769

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

p
C__inference_sampling_layer_call_and_return_conditional_losses_95857

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
random_normal/stddevÎ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed2åôÿ2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
:ÿÿÿÿÿÿÿÿÿ12	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mulV
addAddV2mul:z:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ1:ÿÿÿÿÿÿÿÿÿ1:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

^
B__inference_reshape_layer_call_and_return_conditional_losses_96303

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
¾

-__inference_mnist_enc_var_layer_call_fn_96077

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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_960132
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in


,__inference_mnist_ae_var_layer_call_fn_96901

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
identity¢StatefulPartitionedCallÍ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_967932
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
«	
Ì
2__inference_separable_conv2d_1_layer_call_fn_95738

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
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_957262
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
¶û
Ö
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97446

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
identity¢-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp¢,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp¢7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp¢@mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp¢Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1¢7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp¢@mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp¢Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1¢7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp¢@mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp¢Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1¢+mnist_enc_var/conv2d/BiasAdd/ReadVariableOp¢*mnist_enc_var/conv2d/Conv2D/ReadVariableOp¢-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp¢,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp¢*mnist_enc_var/dense/BiasAdd/ReadVariableOp¢)mnist_enc_var/dense/MatMul/ReadVariableOp¢,mnist_enc_var/dense_1/BiasAdd/ReadVariableOp¢+mnist_enc_var/dense_1/MatMul/ReadVariableOp¢5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp¢>mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp¢@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1¢7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp¢@mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp¢Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ô
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpReadVariableOp3mnist_enc_var_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*mnist_enc_var/conv2d/Conv2D/ReadVariableOpâ
mnist_enc_var/conv2d/Conv2DConv2Dinputs2mnist_enc_var/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_enc_var/conv2d/Conv2DË
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpReadVariableOp4mnist_enc_var_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+mnist_enc_var/conv2d/BiasAdd/ReadVariableOpÜ
mnist_enc_var/conv2d/BiasAddBiasAdd$mnist_enc_var/conv2d/Conv2D:output:03mnist_enc_var/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_enc_var/conv2d/BiasAdd
mnist_enc_var/conv2d/SeluSelu%mnist_enc_var/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
@mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1Ç
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            27
5mnist_enc_var/separable_conv2d/separable_conv2d/ShapeÏ
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/separable_conv2d/separable_conv2d/dilation_rateâ
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative'mnist_enc_var/conv2d/Selu:activations:0Fmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2;
9mnist_enc_var/separable_conv2d/separable_conv2d/depthwiseÝ
/mnist_enc_var/separable_conv2d/separable_conv2dConv2DBmnist_enc_var/separable_conv2d/separable_conv2d/depthwise:output:0Hmnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
21
/mnist_enc_var/separable_conv2d/separable_conv2dé
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp>mnist_enc_var_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype027
5mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp
&mnist_enc_var/separable_conv2d/BiasAddBiasAdd8mnist_enc_var/separable_conv2d/separable_conv2d:output:0=mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12(
&mnist_enc_var/separable_conv2d/BiasAdd½
#mnist_enc_var/separable_conv2d/SeluSelu/mnist_enc_var/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12%
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
Bmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ë
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_enc_var/separable_conv2d_1/separable_conv2d/ShapeÓ
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_enc_var/separable_conv2d_1/separable_conv2d/dilation_rateò
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative1mnist_enc_var/separable_conv2d/Selu:activations:0Hmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2=
;mnist_enc_var/separable_conv2d_1/separable_conv2d/depthwiseå
1mnist_enc_var/separable_conv2d_1/separable_conv2dConv2DDmnist_enc_var/separable_conv2d_1/separable_conv2d/depthwise:output:0Jmnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_enc_var/separable_conv2d_1/separable_conv2dï
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@mnist_enc_var_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp
(mnist_enc_var/separable_conv2d_1/BiasAddBiasAdd:mnist_enc_var/separable_conv2d_1/separable_conv2d:output:0?mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_enc_var/separable_conv2d_1/BiasAddÃ
%mnist_enc_var/separable_conv2d_1/SeluSelu1mnist_enc_var/separable_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
%mnist_enc_var/separable_conv2d_1/SeluÚ
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5mnist_enc_var_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:11*
dtype02.
,mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp
mnist_enc_var/conv2d_1/Conv2DConv2D3mnist_enc_var/separable_conv2d_1/Selu:activations:04mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2
mnist_enc_var/conv2d_1/Conv2DÑ
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6mnist_enc_var_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02/
-mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOpä
mnist_enc_var/conv2d_1/BiasAddBiasAdd&mnist_enc_var/conv2d_1/Conv2D:output:05mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12 
mnist_enc_var/conv2d_1/BiasAdd¥
mnist_enc_var/conv2d_1/SeluSelu'mnist_enc_var/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/conv2d_1/SeluÏ
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=mnist_enc_var/global_average_pooling2d/Mean/reduction_indices
+mnist_enc_var/global_average_pooling2d/MeanMean)mnist_enc_var/conv2d_1/Selu:activations:0Fmnist_enc_var/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12-
+mnist_enc_var/global_average_pooling2d/MeanÉ
)mnist_enc_var/dense/MatMul/ReadVariableOpReadVariableOp2mnist_enc_var_dense_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02+
)mnist_enc_var/dense/MatMul/ReadVariableOpÝ
mnist_enc_var/dense/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:01mnist_enc_var/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense/MatMulÈ
*mnist_enc_var/dense/BiasAdd/ReadVariableOpReadVariableOp3mnist_enc_var_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02,
*mnist_enc_var/dense/BiasAdd/ReadVariableOpÑ
mnist_enc_var/dense/BiasAddBiasAdd$mnist_enc_var/dense/MatMul:product:02mnist_enc_var/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense/BiasAdd
mnist_enc_var/dense/SeluSelu$mnist_enc_var/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense/SeluÏ
+mnist_enc_var/dense_1/MatMul/ReadVariableOpReadVariableOp4mnist_enc_var_dense_1_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02-
+mnist_enc_var/dense_1/MatMul/ReadVariableOpã
mnist_enc_var/dense_1/MatMulMatMul4mnist_enc_var/global_average_pooling2d/Mean:output:03mnist_enc_var/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense_1/MatMulÎ
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpReadVariableOp5mnist_enc_var_dense_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02.
,mnist_enc_var/dense_1/BiasAdd/ReadVariableOpÙ
mnist_enc_var/dense_1/BiasAddBiasAdd&mnist_enc_var/dense_1/MatMul:product:04mnist_enc_var/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/dense_1/BiasAdd
mnist_enc_var/dense_1/SeluSelu&mnist_enc_var/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
+mnist_enc_var/sampling/random_normal/stddev
9mnist_enc_var/sampling/random_normal/RandomStandardNormalRandomStandardNormal%mnist_enc_var/sampling/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed22;
9mnist_enc_var/sampling/random_normal/RandomStandardNormal
(mnist_enc_var/sampling/random_normal/mulMulBmnist_enc_var/sampling/random_normal/RandomStandardNormal:output:04mnist_enc_var/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_enc_var/sampling/random_normal/mulé
$mnist_enc_var/sampling/random_normalAddV2,mnist_enc_var/sampling/random_normal/mul:z:02mnist_enc_var/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12&
$mnist_enc_var/sampling/random_normal
 mnist_enc_var/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 mnist_enc_var/sampling/truediv/yÒ
mnist_enc_var/sampling/truedivRealDiv(mnist_enc_var/dense_1/Selu:activations:0)mnist_enc_var/sampling/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12 
mnist_enc_var/sampling/truediv
mnist_enc_var/sampling/ExpExp"mnist_enc_var/sampling/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/sampling/Exp»
mnist_enc_var/sampling/mulMul(mnist_enc_var/sampling/random_normal:z:0mnist_enc_var/sampling/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mnist_enc_var/sampling/mul»
mnist_enc_var/sampling/addAddV2mnist_enc_var/sampling/mul:z:0&mnist_enc_var/dense/Selu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
)mnist_dec_var/reshape/strided_slice/stack¤
+mnist_dec_var/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_1¤
+mnist_dec_var/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+mnist_dec_var/reshape/strided_slice/stack_2æ
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
%mnist_dec_var/reshape/Reshape/shape/3¾
#mnist_dec_var/reshape/Reshape/shapePack,mnist_dec_var/reshape/strided_slice:output:0.mnist_dec_var/reshape/Reshape/shape/1:output:0.mnist_dec_var/reshape/Reshape/shape/2:output:0.mnist_dec_var/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#mnist_dec_var/reshape/Reshape/shapeÑ
mnist_dec_var/reshape/ReshapeReshapemnist_enc_var/sampling/add:z:0,mnist_dec_var/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Bmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ë
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7mnist_dec_var/separable_conv2d_2/separable_conv2d/ShapeÓ
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_2/separable_conv2d/dilation_rateç
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative&mnist_dec_var/reshape/Reshape:output:0Hmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_2/separable_conv2d/depthwiseå
1mnist_dec_var/separable_conv2d_2/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_2/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_2/separable_conv2dï
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_2/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_2/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_dec_var/separable_conv2d_2/BiasAddÃ
%mnist_dec_var/separable_conv2d_2/SeluSelu1mnist_dec_var/separable_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
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
#mnist_dec_var/up_sampling2d/Const_1È
mnist_dec_var/up_sampling2d/mulMul*mnist_dec_var/up_sampling2d/Const:output:0,mnist_dec_var/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2!
mnist_dec_var/up_sampling2d/mul»
8mnist_dec_var/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_2/Selu:activations:0#mnist_dec_var/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
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
Bmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ë
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_3/separable_conv2d/ShapeÓ
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_3/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativeImnist_dec_var/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_3/separable_conv2d/depthwiseå
1mnist_dec_var/separable_conv2d_3/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_3/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_3/separable_conv2dï
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_3/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_3/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_dec_var/separable_conv2d_3/BiasAddÃ
%mnist_dec_var/separable_conv2d_3/SeluSelu1mnist_dec_var/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
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
%mnist_dec_var/up_sampling2d_1/Const_1Ð
!mnist_dec_var/up_sampling2d_1/mulMul,mnist_dec_var/up_sampling2d_1/Const:output:0.mnist_dec_var/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2#
!mnist_dec_var/up_sampling2d_1/mulÁ
:mnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3mnist_dec_var/separable_conv2d_3/Selu:activations:0%mnist_dec_var/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
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
Bmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ë
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      29
7mnist_dec_var/separable_conv2d_4/separable_conv2d/ShapeÓ
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?mnist_dec_var/separable_conv2d_4/separable_conv2d/dilation_rate
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeKmnist_dec_var/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Hmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
2=
;mnist_dec_var/separable_conv2d_4/separable_conv2d/depthwiseå
1mnist_dec_var/separable_conv2d_4/separable_conv2dConv2DDmnist_dec_var/separable_conv2d_4/separable_conv2d/depthwise:output:0Jmnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingVALID*
strides
23
1mnist_dec_var/separable_conv2d_4/separable_conv2dï
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp@mnist_dec_var_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype029
7mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp
(mnist_dec_var/separable_conv2d_4/BiasAddBiasAdd:mnist_dec_var/separable_conv2d_4/separable_conv2d:output:0?mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12*
(mnist_dec_var/separable_conv2d_4/BiasAddÃ
%mnist_dec_var/separable_conv2d_4/SeluSelu1mnist_dec_var/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12'
%mnist_dec_var/separable_conv2d_4/SeluÚ
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5mnist_dec_var_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype02.
,mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp
mnist_dec_var/conv2d_2/Conv2DConv2D3mnist_dec_var/separable_conv2d_4/Selu:activations:04mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
mnist_dec_var/conv2d_2/Conv2DÑ
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6mnist_dec_var_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOpä
mnist_dec_var/conv2d_2/BiasAddBiasAdd&mnist_dec_var/conv2d_2/Conv2D:output:05mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
mnist_dec_var/conv2d_2/BiasAdd¥
mnist_dec_var/conv2d_2/SeluSelu'mnist_dec_var/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mnist_dec_var/conv2d_2/Selu
IdentityIdentity)mnist_dec_var/conv2d_2/Selu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityá
NoOpNoOp.^mnist_dec_var/conv2d_2/BiasAdd/ReadVariableOp-^mnist_dec_var/conv2d_2/Conv2D/ReadVariableOp8^mnist_dec_var/separable_conv2d_2/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_2/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_3/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_3/separable_conv2d/ReadVariableOp_18^mnist_dec_var/separable_conv2d_4/BiasAdd/ReadVariableOpA^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOpC^mnist_dec_var/separable_conv2d_4/separable_conv2d/ReadVariableOp_1,^mnist_enc_var/conv2d/BiasAdd/ReadVariableOp+^mnist_enc_var/conv2d/Conv2D/ReadVariableOp.^mnist_enc_var/conv2d_1/BiasAdd/ReadVariableOp-^mnist_enc_var/conv2d_1/Conv2D/ReadVariableOp+^mnist_enc_var/dense/BiasAdd/ReadVariableOp*^mnist_enc_var/dense/MatMul/ReadVariableOp-^mnist_enc_var/dense_1/BiasAdd/ReadVariableOp,^mnist_enc_var/dense_1/MatMul/ReadVariableOp6^mnist_enc_var/separable_conv2d/BiasAdd/ReadVariableOp?^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOpA^mnist_enc_var/separable_conv2d/separable_conv2d/ReadVariableOp_18^mnist_enc_var/separable_conv2d_1/BiasAdd/ReadVariableOpA^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOpC^mnist_enc_var/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2^
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»'

H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_96567

dec_in2
separable_conv2d_2_96538:2
separable_conv2d_2_96540:1&
separable_conv2d_2_96542:12
separable_conv2d_3_96546:12
separable_conv2d_3_96548:11&
separable_conv2d_3_96550:12
separable_conv2d_4_96554:12
separable_conv2d_4_96556:11&
separable_conv2d_4_96558:1(
conv2d_2_96561:1
conv2d_2_96563:
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
B__inference_reshape_layer_call_and_return_conditional_losses_963032
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96538separable_conv2d_2_96540separable_conv2d_2_96542*
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
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_961742,
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
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_961992
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96546separable_conv2d_3_96548separable_conv2d_3_96550*
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
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_962222,
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
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_962472!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96554separable_conv2d_4_96556separable_conv2d_4_96558*
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
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_962702,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96561conv2d_2_96563*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_963392"
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
¸
Å	
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_96957

enc_in-
mnist_enc_var_96904:!
mnist_enc_var_96906:-
mnist_enc_var_96908:-
mnist_enc_var_96910:1!
mnist_enc_var_96912:1-
mnist_enc_var_96914:1-
mnist_enc_var_96916:11!
mnist_enc_var_96918:1-
mnist_enc_var_96920:11!
mnist_enc_var_96922:1%
mnist_enc_var_96924:11!
mnist_enc_var_96926:1%
mnist_enc_var_96928:11!
mnist_enc_var_96930:1-
mnist_dec_var_96933:-
mnist_dec_var_96935:1!
mnist_dec_var_96937:1-
mnist_dec_var_96939:1-
mnist_dec_var_96941:11!
mnist_dec_var_96943:1-
mnist_dec_var_96945:1-
mnist_dec_var_96947:11!
mnist_dec_var_96949:1-
mnist_dec_var_96951:1!
mnist_dec_var_96953:
identity¢%mnist_dec_var/StatefulPartitionedCall¢%mnist_enc_var/StatefulPartitionedCallÁ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallenc_inmnist_enc_var_96904mnist_enc_var_96906mnist_enc_var_96908mnist_enc_var_96910mnist_enc_var_96912mnist_enc_var_96914mnist_enc_var_96916mnist_enc_var_96918mnist_enc_var_96920mnist_enc_var_96922mnist_enc_var_96924mnist_enc_var_96926mnist_enc_var_96928mnist_enc_var_96930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_958602'
%mnist_enc_var/StatefulPartitionedCall¾
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_96933mnist_dec_var_96935mnist_dec_var_96937mnist_dec_var_96939mnist_dec_var_96941mnist_dec_var_96943mnist_dec_var_96945mnist_dec_var_96947mnist_dec_var_96949mnist_dec_var_96951mnist_dec_var_96953*
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_963462'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in

ó
B__inference_dense_1_layer_call_and_return_conditional_losses_97914

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Selum
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


,__inference_mnist_ae_var_layer_call_fn_97186

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
identity¢StatefulPartitionedCallÍ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_967932
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

r
C__inference_sampling_layer_call_and_return_conditional_losses_97936
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
random_normal/stddevÍ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0*
seed±ÿå)*
seed2¥2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
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
:ÿÿÿÿÿÿÿÿÿ12	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
mulX
addAddV2mul:z:0inputs_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ1:ÿÿÿÿÿÿÿÿÿ1:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
"
_user_specified_name
inputs/1
µ
T
8__inference_global_average_pooling2d_layer_call_fn_95751

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
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_957452
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
Ô
I
-__inference_up_sampling2d_layer_call_fn_96205

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
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_961992
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
øè
:
!__inference__traced_restore_98500
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
identity_83¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_9Ã,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*Ï+
valueÅ+BÂ+SB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names·
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*»
value±B®SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÍ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*â
_output_shapesÏ
Ì:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
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
Identity_17ª
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
Identity_19¿
AssignVariableOp_19AssignVariableOp7assignvariableop_19_separable_conv2d_2_depthwise_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¿
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
Identity_22¿
AssignVariableOp_22AssignVariableOp7assignvariableop_22_separable_conv2d_3_depthwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¿
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
Identity_25¿
AssignVariableOp_25AssignVariableOp7assignvariableop_25_separable_conv2d_4_depthwise_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¿
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
Identity_30¡
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¡
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
Identity_33®
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_conv2d_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ä
AssignVariableOp_34AssignVariableOp<assignvariableop_34_adam_separable_conv2d_depthwise_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ä
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adam_separable_conv2d_pointwise_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¸
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_separable_conv2d_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Æ
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_separable_conv2d_1_depthwise_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Æ
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_separable_conv2d_1_pointwise_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39º
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
Identity_42¯
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
Identity_45¯
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_1_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Æ
AssignVariableOp_46AssignVariableOp>assignvariableop_46_adam_separable_conv2d_2_depthwise_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Æ
AssignVariableOp_47AssignVariableOp>assignvariableop_47_adam_separable_conv2d_2_pointwise_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48º
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_separable_conv2d_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Æ
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_separable_conv2d_3_depthwise_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Æ
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adam_separable_conv2d_3_pointwise_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51º
AssignVariableOp_51AssignVariableOp2assignvariableop_51_adam_separable_conv2d_3_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Æ
AssignVariableOp_52AssignVariableOp>assignvariableop_52_adam_separable_conv2d_4_depthwise_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Æ
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_separable_conv2d_4_pointwise_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54º
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
Identity_58®
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_conv2d_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ä
AssignVariableOp_59AssignVariableOp<assignvariableop_59_adam_separable_conv2d_depthwise_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ä
AssignVariableOp_60AssignVariableOp<assignvariableop_60_adam_separable_conv2d_pointwise_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¸
AssignVariableOp_61AssignVariableOp0assignvariableop_61_adam_separable_conv2d_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Æ
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_separable_conv2d_1_depthwise_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Æ
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adam_separable_conv2d_1_pointwise_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64º
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
Identity_67¯
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
Identity_70¯
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Æ
AssignVariableOp_71AssignVariableOp>assignvariableop_71_adam_separable_conv2d_2_depthwise_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Æ
AssignVariableOp_72AssignVariableOp>assignvariableop_72_adam_separable_conv2d_2_pointwise_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73º
AssignVariableOp_73AssignVariableOp2assignvariableop_73_adam_separable_conv2d_2_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Æ
AssignVariableOp_74AssignVariableOp>assignvariableop_74_adam_separable_conv2d_3_depthwise_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Æ
AssignVariableOp_75AssignVariableOp>assignvariableop_75_adam_separable_conv2d_3_pointwise_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76º
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_separable_conv2d_3_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Æ
AssignVariableOp_77AssignVariableOp>assignvariableop_77_adam_separable_conv2d_4_depthwise_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Æ
AssignVariableOp_78AssignVariableOp>assignvariableop_78_adam_separable_conv2d_4_pointwise_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79º
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
NoOpê
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82f
Identity_83IdentityIdentity_82:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_83Ò
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
Ë
C
'__inference_reshape_layer_call_fn_97941

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
B__inference_reshape_layer_call_and_return_conditional_losses_963032
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
ç
ú
A__inference_conv2d_layer_call_and_return_conditional_losses_97854

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
¸
Å	
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_96793

inputs-
mnist_enc_var_96740:!
mnist_enc_var_96742:-
mnist_enc_var_96744:-
mnist_enc_var_96746:1!
mnist_enc_var_96748:1-
mnist_enc_var_96750:1-
mnist_enc_var_96752:11!
mnist_enc_var_96754:1-
mnist_enc_var_96756:11!
mnist_enc_var_96758:1%
mnist_enc_var_96760:11!
mnist_enc_var_96762:1%
mnist_enc_var_96764:11!
mnist_enc_var_96766:1-
mnist_dec_var_96769:-
mnist_dec_var_96771:1!
mnist_dec_var_96773:1-
mnist_dec_var_96775:1-
mnist_dec_var_96777:11!
mnist_dec_var_96779:1-
mnist_dec_var_96781:1-
mnist_dec_var_96783:11!
mnist_dec_var_96785:1-
mnist_dec_var_96787:1!
mnist_dec_var_96789:
identity¢%mnist_dec_var/StatefulPartitionedCall¢%mnist_enc_var/StatefulPartitionedCallÁ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallinputsmnist_enc_var_96740mnist_enc_var_96742mnist_enc_var_96744mnist_enc_var_96746mnist_enc_var_96748mnist_enc_var_96750mnist_enc_var_96752mnist_enc_var_96754mnist_enc_var_96756mnist_enc_var_96758mnist_enc_var_96760mnist_enc_var_96762mnist_enc_var_96764mnist_enc_var_96766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_960132'
%mnist_enc_var/StatefulPartitionedCall¾
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_96769mnist_dec_var_96771mnist_dec_var_96773mnist_dec_var_96775mnist_dec_var_96777mnist_dec_var_96779mnist_dec_var_96781mnist_dec_var_96783mnist_dec_var_96785mnist_dec_var_96787mnist_dec_var_96789*
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_964492'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«	
Ì
2__inference_separable_conv2d_4_layer_call_fn_96282

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
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_962702
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


(__inference_conv2d_1_layer_call_fn_97863

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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_958002
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
ã

(__inference_conv2d_2_layer_call_fn_97964

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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_963392
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
»'

H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_96534

dec_in2
separable_conv2d_2_96505:2
separable_conv2d_2_96507:1&
separable_conv2d_2_96509:12
separable_conv2d_3_96513:12
separable_conv2d_3_96515:11&
separable_conv2d_3_96517:12
separable_conv2d_4_96521:12
separable_conv2d_4_96523:11&
separable_conv2d_4_96525:1(
conv2d_2_96528:1
conv2d_2_96530:
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
B__inference_reshape_layer_call_and_return_conditional_losses_963032
reshape/PartitionedCall
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0separable_conv2d_2_96505separable_conv2d_2_96507separable_conv2d_2_96509*
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
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_961742,
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
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_961992
up_sampling2d/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0separable_conv2d_3_96513separable_conv2d_3_96515separable_conv2d_3_96517*
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
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_962222,
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
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_962472!
up_sampling2d_1/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0separable_conv2d_4_96521separable_conv2d_4_96523separable_conv2d_4_96525*
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
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_962702,
*separable_conv2d_4/StatefulPartitionedCallÛ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0conv2d_2_96528conv2d_2_96530*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_963392"
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
¸
Å	
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97013

enc_in-
mnist_enc_var_96960:!
mnist_enc_var_96962:-
mnist_enc_var_96964:-
mnist_enc_var_96966:1!
mnist_enc_var_96968:1-
mnist_enc_var_96970:1-
mnist_enc_var_96972:11!
mnist_enc_var_96974:1-
mnist_enc_var_96976:11!
mnist_enc_var_96978:1%
mnist_enc_var_96980:11!
mnist_enc_var_96982:1%
mnist_enc_var_96984:11!
mnist_enc_var_96986:1-
mnist_dec_var_96989:-
mnist_dec_var_96991:1!
mnist_dec_var_96993:1-
mnist_dec_var_96995:1-
mnist_dec_var_96997:11!
mnist_dec_var_96999:1-
mnist_dec_var_97001:1-
mnist_dec_var_97003:11!
mnist_dec_var_97005:1-
mnist_dec_var_97007:1!
mnist_dec_var_97009:
identity¢%mnist_dec_var/StatefulPartitionedCall¢%mnist_enc_var/StatefulPartitionedCallÁ
%mnist_enc_var/StatefulPartitionedCallStatefulPartitionedCallenc_inmnist_enc_var_96960mnist_enc_var_96962mnist_enc_var_96964mnist_enc_var_96966mnist_enc_var_96968mnist_enc_var_96970mnist_enc_var_96972mnist_enc_var_96974mnist_enc_var_96976mnist_enc_var_96978mnist_enc_var_96980mnist_enc_var_96982mnist_enc_var_96984mnist_enc_var_96986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_960132'
%mnist_enc_var/StatefulPartitionedCall¾
%mnist_dec_var/StatefulPartitionedCallStatefulPartitionedCall.mnist_enc_var/StatefulPartitionedCall:output:0mnist_dec_var_96989mnist_dec_var_96991mnist_dec_var_96993mnist_dec_var_96995mnist_dec_var_96997mnist_dec_var_96999mnist_dec_var_97001mnist_dec_var_97003mnist_dec_var_97005mnist_dec_var_97007mnist_dec_var_97009*
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
GPU2*0J 8 *Q
fLRJ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_964492'
%mnist_dec_var/StatefulPartitionedCall£
IdentityIdentity.mnist_dec_var/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp&^mnist_dec_var/StatefulPartitionedCall&^mnist_enc_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%mnist_dec_var/StatefulPartitionedCall%mnist_dec_var/StatefulPartitionedCall2N
%mnist_enc_var/StatefulPartitionedCall%mnist_enc_var/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameenc_in
ã
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_95745

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
¥-

H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_95860

inputs&
conv2d_95770:
conv2d_95772:0
separable_conv2d_95775:0
separable_conv2d_95777:1$
separable_conv2d_95779:12
separable_conv2d_1_95782:12
separable_conv2d_1_95784:11&
separable_conv2d_1_95786:1(
conv2d_1_95801:11
conv2d_1_95803:1
dense_95819:11
dense_95821:1
dense_1_95836:11
dense_1_95838:1
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ sampling/StatefulPartitionedCall¢(separable_conv2d/StatefulPartitionedCall¢*separable_conv2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_95770conv2d_95772*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_957692 
conv2d/StatefulPartitionedCallÿ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0separable_conv2d_95775separable_conv2d_95777separable_conv2d_95779*
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
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_956972*
(separable_conv2d/StatefulPartitionedCall
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_95782separable_conv2d_1_95784separable_conv2d_1_95786*
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
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_957262,
*separable_conv2d_1/StatefulPartitionedCallÉ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0conv2d_1_95801conv2d_1_95803*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_958002"
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
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_957452*
(global_average_pooling2d/PartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_95819dense_95821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_958182
dense/StatefulPartitionedCallº
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_1_95836dense_1_95838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_958352!
dense_1/StatefulPartitionedCall¹
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_958572"
 sampling/StatefulPartitionedCall
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ12

IdentityÏ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

-__inference_mnist_enc_var_layer_call_fn_97479

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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_958602
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ôh
Ã
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_97834

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
¾

-__inference_mnist_enc_var_layer_call_fn_97512

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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ1*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_960132
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
A
enc_in7
serving_default_enc_in:0ÿÿÿÿÿÿÿÿÿI
mnist_dec_var8
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ê
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
_default_save_signature
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
º
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
trainable_variables
regularization_losses
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_network
ù
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
trainable_variables
regularization_losses
 	variables
!	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_network
Õ
"iter

#beta_1

$beta_2
	%decay
&learning_rate'mÛ(mÜ)mÝ*mÞ+mß,mà-má.mâ/mã0mä1må2mæ3mç4mè5mé6mê7më8mì9mí:mî;mï<mð=mñ>mò?mó'vô(võ)vö*v÷+vø,vù-vú.vû/vü0vý1vþ2vÿ3v4v5v6v7v8v9v:v;v<v=v>v?v"
tf_deprecated_optimizer
Þ
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
Þ
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
Î
@non_trainable_variables
trainable_variables
Alayer_regularization_losses
regularization_losses
	variables
Bmetrics

Clayers
Dlayer_metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
½

'kernel
(bias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
)depthwise_kernel
*pointwise_kernel
+bias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
,depthwise_kernel
-pointwise_kernel
.bias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

/kernel
0bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

1kernel
2bias
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
½

3kernel
4bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"
_tf_keras_layer
§
atrainable_variables
b	variables
cregularization_losses
d	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"
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
 "
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
°
enon_trainable_variables
trainable_variables
flayer_regularization_losses
regularization_losses
	variables
gmetrics

hlayers
ilayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
§
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
5depthwise_kernel
6pointwise_kernel
7bias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
§
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
8depthwise_kernel
9pointwise_kernel
:bias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
§
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
­__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
ß
;depthwise_kernel
<pointwise_kernel
=bias
~trainable_variables
	variables
regularization_losses
	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

>kernel
?bias
trainable_variables
	variables
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
 "
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
µ
non_trainable_variables
trainable_variables
 layer_regularization_losses
regularization_losses
 	variables
metrics
layers
layer_metrics
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
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
µ
non_trainable_variables
Etrainable_variables
 layer_regularization_losses
F	variables
Gregularization_losses
metrics
layers
layer_metrics
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
µ
non_trainable_variables
Itrainable_variables
 layer_regularization_losses
J	variables
Kregularization_losses
metrics
layers
layer_metrics
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
µ
non_trainable_variables
Mtrainable_variables
 layer_regularization_losses
N	variables
Oregularization_losses
metrics
layers
layer_metrics
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
µ
non_trainable_variables
Qtrainable_variables
 layer_regularization_losses
R	variables
Sregularization_losses
metrics
layers
layer_metrics
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
µ
 non_trainable_variables
Utrainable_variables
 ¡layer_regularization_losses
V	variables
Wregularization_losses
¢metrics
£layers
¤layer_metrics
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
µ
¥non_trainable_variables
Ytrainable_variables
 ¦layer_regularization_losses
Z	variables
[regularization_losses
§metrics
¨layers
©layer_metrics
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
µ
ªnon_trainable_variables
]trainable_variables
 «layer_regularization_losses
^	variables
_regularization_losses
¬metrics
­layers
®layer_metrics
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¯non_trainable_variables
atrainable_variables
 °layer_regularization_losses
b	variables
cregularization_losses
±metrics
²layers
³layer_metrics
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
´non_trainable_variables
jtrainable_variables
 µlayer_regularization_losses
k	variables
lregularization_losses
¶metrics
·layers
¸layer_metrics
¥__call__
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
µ
¹non_trainable_variables
ntrainable_variables
 ºlayer_regularization_losses
o	variables
pregularization_losses
»metrics
¼layers
½layer_metrics
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
µ
¾non_trainable_variables
rtrainable_variables
 ¿layer_regularization_losses
s	variables
tregularization_losses
Àmetrics
Álayers
Âlayer_metrics
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
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
µ
Ãnon_trainable_variables
vtrainable_variables
 Älayer_regularization_losses
w	variables
xregularization_losses
Åmetrics
Ælayers
Çlayer_metrics
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
µ
Ènon_trainable_variables
ztrainable_variables
 Élayer_regularization_losses
{	variables
|regularization_losses
Êmetrics
Ëlayers
Ìlayer_metrics
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
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
¶
Ínon_trainable_variables
~trainable_variables
 Îlayer_regularization_losses
	variables
regularization_losses
Ïmetrics
Ðlayers
Ñlayer_metrics
¯__call__
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
¸
Ònon_trainable_variables
trainable_variables
 Ólayer_regularization_losses
	variables
regularization_losses
Ômetrics
Õlayers
Ölayer_metrics
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper
R

×total

Øcount
Ù	variables
Ú	keras_api"
_tf_keras_metric
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
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
×0
Ø1"
trackable_list_wrapper
.
Ù	variables"
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
å2â
 __inference__wrapped_model_95680½
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
þ2û
,__inference_mnist_ae_var_layer_call_fn_96680
,__inference_mnist_ae_var_layer_call_fn_97131
,__inference_mnist_ae_var_layer_call_fn_97186
,__inference_mnist_ae_var_layer_call_fn_96901À
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
ê2ç
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97316
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97446
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_96957
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97013À
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
2ÿ
-__inference_mnist_enc_var_layer_call_fn_95891
-__inference_mnist_enc_var_layer_call_fn_97479
-__inference_mnist_enc_var_layer_call_fn_97512
-__inference_mnist_enc_var_layer_call_fn_96077À
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
î2ë
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_97581
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_97650
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96117
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96157À
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
2ÿ
-__inference_mnist_dec_var_layer_call_fn_96371
-__inference_mnist_dec_var_layer_call_fn_97677
-__inference_mnist_dec_var_layer_call_fn_97704
-__inference_mnist_dec_var_layer_call_fn_96501À
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
î2ë
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_97769
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_97834
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_96534
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_96567À
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
#__inference_signature_wrapper_97076enc_in"
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
Ð2Í
&__inference_conv2d_layer_call_fn_97843¢
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
ë2è
A__inference_conv2d_layer_call_and_return_conditional_losses_97854¢
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
2
0__inference_separable_conv2d_layer_call_fn_95709×
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
ª2§
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95697×
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
2__inference_separable_conv2d_1_layer_call_fn_95738×
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
¬2©
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95726×
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
Ò2Ï
(__inference_conv2d_1_layer_call_fn_97863¢
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
í2ê
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97874¢
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
 2
8__inference_global_average_pooling2d_layer_call_fn_95751à
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
»2¸
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_95745à
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
Ï2Ì
%__inference_dense_layer_call_fn_97883¢
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
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_97894¢
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
'__inference_dense_1_layer_call_fn_97903¢
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
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_97914¢
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
(__inference_sampling_layer_call_fn_97920¢
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
í2ê
C__inference_sampling_layer_call_and_return_conditional_losses_97936¢
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
'__inference_reshape_layer_call_fn_97941¢
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
ì2é
B__inference_reshape_layer_call_and_return_conditional_losses_97955¢
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
2
2__inference_separable_conv2d_2_layer_call_fn_96186×
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
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_96174×
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
2
-__inference_up_sampling2d_layer_call_fn_96205à
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
°2­
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_96199à
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
2
2__inference_separable_conv2d_3_layer_call_fn_96234×
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
¬2©
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_96222×
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
2
/__inference_up_sampling2d_1_layer_call_fn_96253à
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
²2¯
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_96247à
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
2
2__inference_separable_conv2d_4_layer_call_fn_96282×
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
¬2©
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_96270×
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
Ò2Ï
(__inference_conv2d_2_layer_call_fn_97964¢
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
í2ê
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97975¢
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
 À
 __inference__wrapped_model_95680'()*+,-./0123456789:;<=>?7¢4
-¢*
(%
enc_inÿÿÿÿÿÿÿÿÿ
ª "EªB
@
mnist_dec_var/,
mnist_dec_varÿÿÿÿÿÿÿÿÿ³
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97874l/07¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_conv2d_1_layer_call_fn_97863_/07¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1Ø
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97975>?I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
(__inference_conv2d_2_layer_call_fn_97964>?I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
A__inference_conv2d_layer_call_and_return_conditional_losses_97854l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_conv2d_layer_call_fn_97843_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_1_layer_call_and_return_conditional_losses_97914\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 z
'__inference_dense_1_layer_call_fn_97903O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1 
@__inference_dense_layer_call_and_return_conditional_losses_97894\12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 x
%__inference_dense_layer_call_fn_97883O12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1Ü
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_95745R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
8__inference_global_average_pooling2d_layer_call_fn_95751wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_96957'()*+,-./0123456789:;<=>??¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 é
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97013'()*+,-./0123456789:;<=>??¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ×
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97316'()*+,-./0123456789:;<=>??¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ×
G__inference_mnist_ae_var_layer_call_and_return_conditional_losses_97446'()*+,-./0123456789:;<=>??¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
,__inference_mnist_ae_var_layer_call_fn_96680'()*+,-./0123456789:;<=>??¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
,__inference_mnist_ae_var_layer_call_fn_96901'()*+,-./0123456789:;<=>??¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
,__inference_mnist_ae_var_layer_call_fn_97131'()*+,-./0123456789:;<=>??¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
,__inference_mnist_ae_var_layer_call_fn_97186'()*+,-./0123456789:;<=>??¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_9653456789:;<=>?7¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_9656756789:;<=>?7¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_97769u56789:;<=>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
H__inference_mnist_dec_var_layer_call_and_return_conditional_losses_97834u56789:;<=>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 «
-__inference_mnist_dec_var_layer_call_fn_96371z56789:;<=>?7¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
-__inference_mnist_dec_var_layer_call_fn_96501z56789:;<=>?7¢4
-¢*
 
dec_inÿÿÿÿÿÿÿÿÿ1
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
-__inference_mnist_dec_var_layer_call_fn_97677z56789:;<=>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
-__inference_mnist_dec_var_layer_call_fn_97704z56789:;<=>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ1
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96117x'()*+,-./01234?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 Ä
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_96157x'()*+,-./01234?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 Ä
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_97581x'()*+,-./01234?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 Ä
H__inference_mnist_enc_var_layer_call_and_return_conditional_losses_97650x'()*+,-./01234?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 
-__inference_mnist_enc_var_layer_call_fn_95891k'()*+,-./01234?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ1
-__inference_mnist_enc_var_layer_call_fn_96077k'()*+,-./01234?¢<
5¢2
(%
enc_inÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ1
-__inference_mnist_enc_var_layer_call_fn_97479k'()*+,-./01234?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ1
-__inference_mnist_enc_var_layer_call_fn_97512k'()*+,-./01234?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ1¦
B__inference_reshape_layer_call_and_return_conditional_losses_97955`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ~
'__inference_reshape_layer_call_fn_97941S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿË
C__inference_sampling_layer_call_and_return_conditional_losses_97936Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ1
"
inputs/1ÿÿÿÿÿÿÿÿÿ1
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 ¢
(__inference_sampling_layer_call_fn_97920vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ1
"
inputs/1ÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_95726,-.I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_1_layer_call_fn_95738,-.I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_96174567I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_2_layer_call_fn_96186567I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_9622289:I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_3_layer_call_fn_9623489:I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1ã
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_96270;<=I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 »
2__inference_separable_conv2d_4_layer_call_fn_96282;<=I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1á
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_95697)*+I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1
 ¹
0__inference_separable_conv2d_layer_call_fn_95709)*+I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ1Í
#__inference_signature_wrapper_97076¥'()*+,-./0123456789:;<=>?A¢>
¢ 
7ª4
2
enc_in(%
enc_inÿÿÿÿÿÿÿÿÿ"EªB
@
mnist_dec_var/,
mnist_dec_varÿÿÿÿÿÿÿÿÿí
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_96247R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_up_sampling2d_1_layer_call_fn_96253R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_96199R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_up_sampling2d_layer_call_fn_96205R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
σΩ
Ν
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
₯
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Α
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
executor_typestring ¨
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
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*	2.8.0-rc02v1.12.1-69099-g0d00ab2fd158ΖΨ

word_embedding_layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
€d*0
shared_name!word_embedding_layer/embeddings

3word_embedding_layer/embeddings/Read/ReadVariableOpReadVariableOpword_embedding_layer/embeddings* 
_output_shapes
:
€d*
dtype0

hidden_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Έd*$
shared_namehidden_layer/kernel
|
'hidden_layer/kernel/Read/ReadVariableOpReadVariableOphidden_layer/kernel*
_output_shapes
:	Έd*
dtype0
z
hidden_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namehidden_layer/bias
s
%hidden_layer/bias/Read/ReadVariableOpReadVariableOphidden_layer/bias*
_output_shapes
:d*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:d*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/hidden_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Έd*+
shared_nameAdam/hidden_layer/kernel/m

.Adam/hidden_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer/kernel/m*
_output_shapes
:	Έd*
dtype0

Adam/hidden_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/hidden_layer/bias/m

,Adam/hidden_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer/bias/m*
_output_shapes
:d*
dtype0

Adam/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/output_layer/kernel/m

.Adam/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/m*
_output_shapes

:d*
dtype0

Adam/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/m

,Adam/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/m*
_output_shapes
:*
dtype0

Adam/hidden_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Έd*+
shared_nameAdam/hidden_layer/kernel/v

.Adam/hidden_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer/kernel/v*
_output_shapes
:	Έd*
dtype0

Adam/hidden_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/hidden_layer/bias/v

,Adam/hidden_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer/bias/v*
_output_shapes
:d*
dtype0

Adam/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/output_layer/kernel/v

.Adam/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/v*
_output_shapes

:d*
dtype0

Adam/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/v

,Adam/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
£0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ή/
valueΤ/BΡ/ BΚ/
Ϋ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
₯
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses* 
¦

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*

3iter

4beta_1

5beta_2
	6decay
7learning_ratemcmd+me,mfvgvh+vi,vj*
'
0
1
2
+3
,4*
 
0
1
+2
,3*
	
80* 
°
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

>serving_default* 
sm
VARIABLE_VALUEword_embedding_layer/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*
* 
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
c]
VARIABLE_VALUEhidden_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhidden_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
	
80* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
'
0
1
2
3
4*

X0
Y1*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
80* 
* 
8
	Ztotal
	[count
\	variables
]	keras_api*
H
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

\	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

^0
_1*

a	variables*

VARIABLE_VALUEAdam/hidden_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hidden_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/output_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/output_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hidden_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hidden_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/output_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/output_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

*serving_default_word_embedding_layer_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
Α
StatefulPartitionedCallStatefulPartitionedCall*serving_default_word_embedding_layer_inputword_embedding_layer/embeddingshidden_layer/kernelhidden_layer/biasoutput_layer/kerneloutput_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1385425
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3word_embedding_layer/embeddings/Read/ReadVariableOp'hidden_layer/kernel/Read/ReadVariableOp%hidden_layer/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/hidden_layer/kernel/m/Read/ReadVariableOp,Adam/hidden_layer/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp.Adam/hidden_layer/kernel/v/Read/ReadVariableOp,Adam/hidden_layer/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1385632
Ϋ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameword_embedding_layer/embeddingshidden_layer/kernelhidden_layer/biasoutput_layer/kerneloutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hidden_layer/kernel/mAdam/hidden_layer/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/mAdam/hidden_layer/kernel/vAdam/hidden_layer/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1385708υι
Ο
ρ
.__inference_sequential_2_layer_call_fn_1385320

inputs
unknown:
€d
	unknown_0:	Έd
	unknown_1:d
	unknown_2:d
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
#
 
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385293
word_embedding_layer_input0
word_embedding_layer_1385271:
€d'
hidden_layer_1385275:	Έd"
hidden_layer_1385277:d&
output_layer_1385281:d"
output_layer_1385283:
identity’!dropout_2/StatefulPartitionedCall’$hidden_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall’5output_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputword_embedding_layer_1385271*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385037λ
flatten_2/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Έ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385047
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hidden_layer_1385275hidden_layer_1385277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385060ς
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385146§
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0output_layer_1385281output_layer_1385283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1385090
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_1385281*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp"^dropout_2/StatefulPartitionedCall%^hidden_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nameword_embedding_layer_input
Μ

.__inference_output_layer_layer_call_fn_1385515

inputs
unknown:d
	unknown_0:
identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1385090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
4
Ε	
 __inference__traced_save_1385632
file_prefix>
:savev2_word_embedding_layer_embeddings_read_readvariableop2
.savev2_hidden_layer_kernel_read_readvariableop0
,savev2_hidden_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_hidden_layer_kernel_m_read_readvariableop7
3savev2_adam_hidden_layer_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop9
5savev2_adam_hidden_layer_kernel_v_read_readvariableop7
3savev2_adam_hidden_layer_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: χ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B Ζ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_word_embedding_layer_embeddings_read_readvariableop.savev2_hidden_layer_kernel_read_readvariableop,savev2_hidden_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_hidden_layer_kernel_m_read_readvariableop3savev2_adam_hidden_layer_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop5savev2_adam_hidden_layer_kernel_v_read_readvariableop3savev2_adam_hidden_layer_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: :
€d:	Έd:d:d:: : : : : : : : : :	Έd:d:d::	Έd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
€d:%!

_output_shapes
:	Έd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Έd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	Έd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 

Ί
__inference_loss_fn_0_1385543P
>output_layer_kernel_regularizer_square_readvariableop_resource:d
identity’5output_layer/kernel/Regularizer/Square/ReadVariableOp΄
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>output_layer_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'output_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp
Ρ0

I__inference_sequential_2_layer_call_and_return_conditional_losses_1385408

inputsA
-word_embedding_layer_embedding_lookup_1385372:
€d>
+hidden_layer_matmul_readvariableop_resource:	Έd:
,hidden_layer_biasadd_readvariableop_resource:d=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identity’#hidden_layer/BiasAdd/ReadVariableOp’"hidden_layer/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp’5output_layer/kernel/Regularizer/Square/ReadVariableOp’%word_embedding_layer/embedding_lookupj
word_embedding_layer/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
%word_embedding_layer/embedding_lookupResourceGather-word_embedding_layer_embedding_lookup_1385372word_embedding_layer/Cast:y:0*
Tindices0*@
_class6
42loc:@word_embedding_layer/embedding_lookup/1385372*+
_output_shapes
:?????????d*
dtype0β
.word_embedding_layer/embedding_lookup/IdentityIdentity.word_embedding_layer/embedding_lookup:output:0*
T0*@
_class6
42loc:@word_embedding_layer/embedding_lookup/1385372*+
_output_shapes
:?????????d«
0word_embedding_layer/embedding_lookup/Identity_1Identity7word_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????Έ  €
flatten_2/ReshapeReshape9word_embedding_layer/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????Έ
"hidden_layer/MatMul/ReadVariableOpReadVariableOp+hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	Έd*
dtype0
hidden_layer/MatMulMatMulflatten_2/Reshape:output:0*hidden_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
#hidden_layer/BiasAdd/ReadVariableOpReadVariableOp,hidden_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
hidden_layer/BiasAddBiasAddhidden_layer/MatMul:product:0+hidden_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dj
hidden_layer/TanhTanhhidden_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ψ;¦?
dropout_2/dropout/MulMulhidden_layer/Tanh:y:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d\
dropout_2/dropout/ShapeShapehidden_layer/Tanh:y:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *k>Δ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
output_layer/MatMulMatMuldropout_2/dropout/Mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????‘
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ό
NoOpNoOp$^hidden_layer/BiasAdd/ReadVariableOp#^hidden_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp&^word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2J
#hidden_layer/BiasAdd/ReadVariableOp#hidden_layer/BiasAdd/ReadVariableOp2H
"hidden_layer/MatMul/ReadVariableOp"hidden_layer/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2N
%word_embedding_layer/embedding_lookup%word_embedding_layer/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
­'
Ο
"__inference__wrapped_model_1385020
word_embedding_layer_inputN
:sequential_2_word_embedding_layer_embedding_lookup_1384997:
€dK
8sequential_2_hidden_layer_matmul_readvariableop_resource:	ΈdG
9sequential_2_hidden_layer_biasadd_readvariableop_resource:dJ
8sequential_2_output_layer_matmul_readvariableop_resource:dG
9sequential_2_output_layer_biasadd_readvariableop_resource:
identity’0sequential_2/hidden_layer/BiasAdd/ReadVariableOp’/sequential_2/hidden_layer/MatMul/ReadVariableOp’0sequential_2/output_layer/BiasAdd/ReadVariableOp’/sequential_2/output_layer/MatMul/ReadVariableOp’2sequential_2/word_embedding_layer/embedding_lookup
&sequential_2/word_embedding_layer/CastCastword_embedding_layer_input*

DstT0*

SrcT0*'
_output_shapes
:?????????Ε
2sequential_2/word_embedding_layer/embedding_lookupResourceGather:sequential_2_word_embedding_layer_embedding_lookup_1384997*sequential_2/word_embedding_layer/Cast:y:0*
Tindices0*M
_classC
A?loc:@sequential_2/word_embedding_layer/embedding_lookup/1384997*+
_output_shapes
:?????????d*
dtype0
;sequential_2/word_embedding_layer/embedding_lookup/IdentityIdentity;sequential_2/word_embedding_layer/embedding_lookup:output:0*
T0*M
_classC
A?loc:@sequential_2/word_embedding_layer/embedding_lookup/1384997*+
_output_shapes
:?????????dΕ
=sequential_2/word_embedding_layer/embedding_lookup/Identity_1IdentityDsequential_2/word_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????dm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????Έ  Λ
sequential_2/flatten_2/ReshapeReshapeFsequential_2/word_embedding_layer/embedding_lookup/Identity_1:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????Έ©
/sequential_2/hidden_layer/MatMul/ReadVariableOpReadVariableOp8sequential_2_hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	Έd*
dtype0Ύ
 sequential_2/hidden_layer/MatMulMatMul'sequential_2/flatten_2/Reshape:output:07sequential_2/hidden_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d¦
0sequential_2/hidden_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_2_hidden_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Δ
!sequential_2/hidden_layer/BiasAddBiasAdd*sequential_2/hidden_layer/MatMul:product:08sequential_2/hidden_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
sequential_2/hidden_layer/TanhTanh*sequential_2/hidden_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
sequential_2/dropout_2/IdentityIdentity"sequential_2/hidden_layer/Tanh:y:0*
T0*'
_output_shapes
:?????????d¨
/sequential_2/output_layer/MatMul/ReadVariableOpReadVariableOp8sequential_2_output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ώ
 sequential_2/output_layer/MatMulMatMul(sequential_2/dropout_2/Identity:output:07sequential_2/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¦
0sequential_2/output_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_2_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Δ
!sequential_2/output_layer/BiasAddBiasAdd*sequential_2/output_layer/MatMul:product:08sequential_2/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
!sequential_2/output_layer/SoftmaxSoftmax*sequential_2/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
IdentityIdentity+sequential_2/output_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ε
NoOpNoOp1^sequential_2/hidden_layer/BiasAdd/ReadVariableOp0^sequential_2/hidden_layer/MatMul/ReadVariableOp1^sequential_2/output_layer/BiasAdd/ReadVariableOp0^sequential_2/output_layer/MatMul/ReadVariableOp3^sequential_2/word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2d
0sequential_2/hidden_layer/BiasAdd/ReadVariableOp0sequential_2/hidden_layer/BiasAdd/ReadVariableOp2b
/sequential_2/hidden_layer/MatMul/ReadVariableOp/sequential_2/hidden_layer/MatMul/ReadVariableOp2d
0sequential_2/output_layer/BiasAdd/ReadVariableOp0sequential_2/output_layer/BiasAdd/ReadVariableOp2b
/sequential_2/output_layer/MatMul/ReadVariableOp/sequential_2/output_layer/MatMul/ReadVariableOp2h
2sequential_2/word_embedding_layer/embedding_lookup2sequential_2/word_embedding_layer/embedding_lookup:c _
'
_output_shapes
:?????????
4
_user_specified_nameword_embedding_layer_input
 !
θ
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385103

inputs0
word_embedding_layer_1385038:
€d'
hidden_layer_1385061:	Έd"
hidden_layer_1385063:d&
output_layer_1385091:d"
output_layer_1385093:
identity’$hidden_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall’5output_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallinputsword_embedding_layer_1385038*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385037λ
flatten_2/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Έ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385047
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hidden_layer_1385061hidden_layer_1385063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385060β
dropout_2/PartitionedCallPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385071
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0output_layer_1385091output_layer_1385093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1385090
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_1385091*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ϋ
NoOpNoOp%^hidden_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ΐ
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385453

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????Έ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????ΈY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????Έ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ω
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385488

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs


ϋ
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385060

inputs1
matmul_readvariableop_resource:	Έd-
biasadd_readvariableop_resource:d
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Έd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Έ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Έ
 
_user_specified_nameinputs
‘
G
+__inference_dropout_2_layer_call_fn_1385478

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385071`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
³	
±
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385037

inputs,
embedding_lookup_1385031:
€d
identity’embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????½
embedding_lookupResourceGatherembedding_lookup_1385031Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/1385031*+
_output_shapes
:?????????d*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/1385031*+
_output_shapes
:?????????d
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ύ

6__inference_word_embedding_layer_layer_call_fn_1385432

inputs
unknown:
€d
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385037s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
τ	
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385146

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ψ;¦?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *k>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ο

.__inference_hidden_layer_layer_call_fn_1385462

inputs
unknown:	Έd
	unknown_0:d
identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385060o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Έ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????Έ
 
_user_specified_nameinputs
τ	
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385500

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ψ;¦?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *k>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ο
ρ
.__inference_sequential_2_layer_call_fn_1385335

inputs
unknown:
€d
	unknown_0:	Έd
	unknown_1:d
	unknown_2:d
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ά!
ό
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385268
word_embedding_layer_input0
word_embedding_layer_1385246:
€d'
hidden_layer_1385250:	Έd"
hidden_layer_1385252:d&
output_layer_1385256:d"
output_layer_1385258:
identity’$hidden_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall’5output_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputword_embedding_layer_1385246*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385037λ
flatten_2/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Έ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385047
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hidden_layer_1385250hidden_layer_1385252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385060β
dropout_2/PartitionedCallPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385071
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0output_layer_1385256output_layer_1385258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1385090
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_1385256*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ϋ
NoOpNoOp%^hidden_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nameword_embedding_layer_input
ς
²
I__inference_output_layer_layer_call_and_return_conditional_losses_1385532

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????―
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
ς
²
I__inference_output_layer_layer_call_and_return_conditional_losses_1385090

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????―
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
	

.__inference_sequential_2_layer_call_fn_1385243
word_embedding_layer_input
unknown:
€d
	unknown_0:	Έd
	unknown_1:d
	unknown_2:d
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nameword_embedding_layer_input
ΐ
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385047

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????Έ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????ΈY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????Έ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ϋ
ό
%__inference_signature_wrapper_1385425
word_embedding_layer_input
unknown:
€d
	unknown_0:	Έd
	unknown_1:d
	unknown_2:d
	unknown_3:
identity’StatefulPartitionedCallς
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1385020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nameword_embedding_layer_input
³	
±
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385442

inputs,
embedding_lookup_1385436:
€d
identity’embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????½
embedding_lookupResourceGatherembedding_lookup_1385436Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/1385436*+
_output_shapes
:?????????d*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/1385436*+
_output_shapes
:?????????d
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
«
G
+__inference_flatten_2_layer_call_fn_1385447

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Έ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385047a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????Έ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
Θ"

I__inference_sequential_2_layer_call_and_return_conditional_losses_1385215

inputs0
word_embedding_layer_1385193:
€d'
hidden_layer_1385197:	Έd"
hidden_layer_1385199:d&
output_layer_1385203:d"
output_layer_1385205:
identity’!dropout_2/StatefulPartitionedCall’$hidden_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall’5output_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallinputsword_embedding_layer_1385193*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385037λ
flatten_2/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Έ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385047
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hidden_layer_1385197hidden_layer_1385199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385060ς
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385146§
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0output_layer_1385203output_layer_1385205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1385090
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_1385203*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp"^dropout_2/StatefulPartitionedCall%^hidden_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
	

.__inference_sequential_2_layer_call_fn_1385116
word_embedding_layer_input
unknown:
€d
	unknown_0:	Έd
	unknown_1:d
	unknown_2:d
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nameword_embedding_layer_input
)

I__inference_sequential_2_layer_call_and_return_conditional_losses_1385368

inputsA
-word_embedding_layer_embedding_lookup_1385339:
€d>
+hidden_layer_matmul_readvariableop_resource:	Έd:
,hidden_layer_biasadd_readvariableop_resource:d=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identity’#hidden_layer/BiasAdd/ReadVariableOp’"hidden_layer/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp’5output_layer/kernel/Regularizer/Square/ReadVariableOp’%word_embedding_layer/embedding_lookupj
word_embedding_layer/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
%word_embedding_layer/embedding_lookupResourceGather-word_embedding_layer_embedding_lookup_1385339word_embedding_layer/Cast:y:0*
Tindices0*@
_class6
42loc:@word_embedding_layer/embedding_lookup/1385339*+
_output_shapes
:?????????d*
dtype0β
.word_embedding_layer/embedding_lookup/IdentityIdentity.word_embedding_layer/embedding_lookup:output:0*
T0*@
_class6
42loc:@word_embedding_layer/embedding_lookup/1385339*+
_output_shapes
:?????????d«
0word_embedding_layer/embedding_lookup/Identity_1Identity7word_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????Έ  €
flatten_2/ReshapeReshape9word_embedding_layer/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????Έ
"hidden_layer/MatMul/ReadVariableOpReadVariableOp+hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	Έd*
dtype0
hidden_layer/MatMulMatMulflatten_2/Reshape:output:0*hidden_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
#hidden_layer/BiasAdd/ReadVariableOpReadVariableOp,hidden_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
hidden_layer/BiasAddBiasAddhidden_layer/MatMul:product:0+hidden_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dj
hidden_layer/TanhTanhhidden_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????dg
dropout_2/IdentityIdentityhidden_layer/Tanh:y:0*
T0*'
_output_shapes
:?????????d
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
output_layer/MatMulMatMuldropout_2/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????‘
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dv
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#output_layer/kernel/Regularizer/SumSum*output_layer/kernel/Regularizer/Square:y:0.output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ό
NoOpNoOp$^hidden_layer/BiasAdd/ReadVariableOp#^hidden_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp&^word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2J
#hidden_layer/BiasAdd/ReadVariableOp#hidden_layer/BiasAdd/ReadVariableOp2H
"hidden_layer/MatMul/ReadVariableOp"hidden_layer/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2N
%word_embedding_layer/embedding_lookup%word_embedding_layer/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
σ
d
+__inference_dropout_2_layer_call_fn_1385483

inputs
identity’StatefulPartitionedCallΑ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ω
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385071

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
ςY
­
#__inference__traced_restore_1385708
file_prefixD
0assignvariableop_word_embedding_layer_embeddings:
€d9
&assignvariableop_1_hidden_layer_kernel:	Έd2
$assignvariableop_2_hidden_layer_bias:d8
&assignvariableop_3_output_layer_kernel:d2
$assignvariableop_4_output_layer_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: A
.assignvariableop_14_adam_hidden_layer_kernel_m:	Έd:
,assignvariableop_15_adam_hidden_layer_bias_m:d@
.assignvariableop_16_adam_output_layer_kernel_m:d:
,assignvariableop_17_adam_output_layer_bias_m:A
.assignvariableop_18_adam_hidden_layer_kernel_v:	Έd:
,assignvariableop_19_adam_hidden_layer_bias_v:d@
.assignvariableop_20_adam_output_layer_kernel_v:d:
,assignvariableop_21_adam_output_layer_bias_v:
identity_23’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ϊ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp0assignvariableop_word_embedding_layer_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_hidden_layer_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_output_layer_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_output_layer_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_hidden_layer_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_hidden_layer_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_output_layer_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_output_layer_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_hidden_layer_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_hidden_layer_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_output_layer_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_output_layer_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
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


ϋ
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385473

inputs1
matmul_readvariableop_resource:	Έd-
biasadd_readvariableop_resource:d
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Έd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Έ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Έ
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Υ
serving_defaultΑ
a
word_embedding_layer_inputC
,serving_default_word_embedding_layer_input:0?????????@
output_layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Οm
υ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
΅

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
Ό
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
»

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer

3iter

4beta_1

5beta_2
	6decay
7learning_ratemcmd+me,mfvgvh+vi,vj"
	optimizer
C
0
1
2
+3
,4"
trackable_list_wrapper
<
0
1
+2
,3"
trackable_list_wrapper
'
80"
trackable_list_wrapper
Κ
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_2_layer_call_fn_1385116
.__inference_sequential_2_layer_call_fn_1385320
.__inference_sequential_2_layer_call_fn_1385335
.__inference_sequential_2_layer_call_fn_1385243ΐ
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385368
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385408
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385268
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385293ΐ
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
ΰBέ
"__inference__wrapped_model_1385020word_embedding_layer_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
>serving_default"
signature_map
3:1
€d2word_embedding_layer/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ΰ2έ
6__inference_word_embedding_layer_layer_call_fn_1385432’
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
ϋ2ψ
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385442’
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_flatten_2_layer_call_fn_1385447’
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
π2ν
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385453’
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
&:$	Έd2hidden_layer/kernel
:d2hidden_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ψ2Υ
.__inference_hidden_layer_layer_call_fn_1385462’
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
σ2π
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385473’
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_2_layer_call_fn_1385478
+__inference_dropout_2_layer_call_fn_1385483΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Κ2Η
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385488
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385500΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
%:#d2output_layer/kernel
:2output_layer/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
'
80"
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
Ψ2Υ
.__inference_output_layer_layer_call_fn_1385515’
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
σ2π
I__inference_output_layer_layer_call_and_return_conditional_losses_1385532’
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
΄2±
__inference_loss_fn_0_1385543
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *’ 
'
0"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
%__inference_signature_wrapper_1385425word_embedding_layer_input"
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
'
0"
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
'
80"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ztotal
	[count
\	variables
]	keras_api"
_tf_keras_metric
^
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
+:)	Έd2Adam/hidden_layer/kernel/m
$:"d2Adam/hidden_layer/bias/m
*:(d2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
+:)	Έd2Adam/hidden_layer/kernel/v
$:"d2Adam/hidden_layer/bias/v
*:(d2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/v°
"__inference__wrapped_model_1385020+,C’@
9’6
41
word_embedding_layer_input?????????
ͺ ";ͺ8
6
output_layer&#
output_layer?????????¦
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385488\3’0
)’&
 
inputs?????????d
p 
ͺ "%’"

0?????????d
 ¦
F__inference_dropout_2_layer_call_and_return_conditional_losses_1385500\3’0
)’&
 
inputs?????????d
p
ͺ "%’"

0?????????d
 ~
+__inference_dropout_2_layer_call_fn_1385478O3’0
)’&
 
inputs?????????d
p 
ͺ "?????????d~
+__inference_dropout_2_layer_call_fn_1385483O3’0
)’&
 
inputs?????????d
p
ͺ "?????????d§
F__inference_flatten_2_layer_call_and_return_conditional_losses_1385453]3’0
)’&
$!
inputs?????????d
ͺ "&’#

0?????????Έ
 
+__inference_flatten_2_layer_call_fn_1385447P3’0
)’&
$!
inputs?????????d
ͺ "?????????Έͺ
I__inference_hidden_layer_layer_call_and_return_conditional_losses_1385473]0’-
&’#
!
inputs?????????Έ
ͺ "%’"

0?????????d
 
.__inference_hidden_layer_layer_call_fn_1385462P0’-
&’#
!
inputs?????????Έ
ͺ "?????????d<
__inference_loss_fn_0_1385543+’

’ 
ͺ " ©
I__inference_output_layer_layer_call_and_return_conditional_losses_1385532\+,/’,
%’"
 
inputs?????????d
ͺ "%’"

0?????????
 
.__inference_output_layer_layer_call_fn_1385515O+,/’,
%’"
 
inputs?????????d
ͺ "?????????Θ
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385268{+,K’H
A’>
41
word_embedding_layer_input?????????
p 

 
ͺ "%’"

0?????????
 Θ
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385293{+,K’H
A’>
41
word_embedding_layer_input?????????
p

 
ͺ "%’"

0?????????
 ΄
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385368g+,7’4
-’*
 
inputs?????????
p 

 
ͺ "%’"

0?????????
 ΄
I__inference_sequential_2_layer_call_and_return_conditional_losses_1385408g+,7’4
-’*
 
inputs?????????
p

 
ͺ "%’"

0?????????
  
.__inference_sequential_2_layer_call_fn_1385116n+,K’H
A’>
41
word_embedding_layer_input?????????
p 

 
ͺ "????????? 
.__inference_sequential_2_layer_call_fn_1385243n+,K’H
A’>
41
word_embedding_layer_input?????????
p

 
ͺ "?????????
.__inference_sequential_2_layer_call_fn_1385320Z+,7’4
-’*
 
inputs?????????
p 

 
ͺ "?????????
.__inference_sequential_2_layer_call_fn_1385335Z+,7’4
-’*
 
inputs?????????
p

 
ͺ "?????????Ρ
%__inference_signature_wrapper_1385425§+,a’^
’ 
WͺT
R
word_embedding_layer_input41
word_embedding_layer_input?????????";ͺ8
6
output_layer&#
output_layer?????????΄
Q__inference_word_embedding_layer_layer_call_and_return_conditional_losses_1385442_/’,
%’"
 
inputs?????????
ͺ ")’&

0?????????d
 
6__inference_word_embedding_layer_layer_call_fn_1385432R/’,
%’"
 
inputs?????????
ͺ "?????????d
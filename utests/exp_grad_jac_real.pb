
8
input_0Placeholder*
dtype0*
shape
:

ExpExpinput_0*
T0

output_0ConjExp*
T0
+
RealRealoutput_0*
T0*

Tout0
D
gradients/ShapeConst*
dtype0*
valueB"      
F
gradients/grad_ys_0/ConstConst*
dtype0*
valueB
 *  �?
b
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
T0*

index_type0
F
gradients/Real_grad/ConstConst*
dtype0*
valueB
 *    
k
gradients/Real_grad/ComplexComplexgradients/grad_ys_0gradients/Real_grad/Const*
T0*

Tout0
J
gradients/output_0_grad/ConjConjgradients/Real_grad/Complex*
T0
L
gradients/Exp_grad/ConjConjExp^gradients/output_0_grad/Conj*
T0
]
gradients/Exp_grad/mulMulgradients/output_0_grad/Conjgradients/Exp_grad/Conj*
T0
L
Reshape/tensorPackgradients/Exp_grad/mul*
N*
T0*

axis 
B
Reshape/shapeConst*
dtype0*
valueB"      
H
ReshapeReshapeReshape/tensorReshape/shape*
T0*
Tshape0
+
ImagImagoutput_0*
T0*

Tout0
F
gradients_1/ShapeConst*
dtype0*
valueB"      
H
gradients_1/grad_ys_0/ConstConst*
dtype0*
valueB
 *  �?
h
gradients_1/grad_ys_0Fillgradients_1/Shapegradients_1/grad_ys_0/Const*
T0*

index_type0
H
gradients_1/Imag_grad/ConstConst*
dtype0*
valueB
 *    
q
gradients_1/Imag_grad/ComplexComplexgradients_1/Imag_grad/Constgradients_1/grad_ys_0*
T0*

Tout0
N
gradients_1/output_0_grad/ConjConjgradients_1/Imag_grad/Complex*
T0
P
gradients_1/Exp_grad/ConjConjExp^gradients_1/output_0_grad/Conj*
T0
c
gradients_1/Exp_grad/mulMulgradients_1/output_0_grad/Conjgradients_1/Exp_grad/Conj*
T0
P
Reshape_1/tensorPackgradients_1/Exp_grad/mul*
N*
T0*

axis 
D
Reshape_1/shapeConst*
dtype0*
valueB"      
N
	Reshape_1ReshapeReshape_1/tensorReshape_1/shape*
T0*
Tshape0
K
jacobian_real_0_0PackReshape	Reshape_1*
N*
T0*

axis
:
	grad_ys_0Placeholder*
dtype0*
shape
:
5
gradients_2/grad_ys_0Identity	grad_ys_0*
T0
F
gradients_2/output_0_grad/ConjConjgradients_2/grad_ys_0*
T0
P
gradients_2/Exp_grad/ConjConjExp^gradients_2/output_0_grad/Conj*
T0
c
gradients_2/Exp_grad/mulMulgradients_2/output_0_grad/Conjgradients_2/Exp_grad/Conj*
T0
O
grad_0_0/tensorPackgradients_2/Exp_grad/mul*
N*
T0*

axis 
C
grad_0_0/shapeConst*
dtype0*
valueB"      
K
grad_0_0Reshapegrad_0_0/tensorgrad_0_0/shape*
T0*
Tshape0"�
# Modeling the PINN

## General Equations

### Gas Channel
The pressure distribution and flow dynamics are described by the continuity equation and the momentum balance:
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\nabla\cdot\boldsymbol{v}&=0\\\rho\,\boldsymbol{v}\cdot\nabla\boldsymbol{v}&=-\nabla&space;p&plus;\mu\,\nabla^2\boldsymbol{v}\end{align}">
</p>

Here, $\nabla$ denotes the Nabla operator, $\rho$ the density of the ideal gas mixture, $\boldsymbol{v} = (u, v, w)^\top$
the velocity vector, $p$ the pressure, and $\mu$ the dynamic viscosity.
Due to the stationary operating point in the CFD model, the temporal dependencies of the continuity and momentum
equation have already been omitted and will be in further equations.

While the fluids flow through the Gas Channel, they diffuse through their Gas Diffusion Layer and Microporous Layer
toward the Catalyst Layer. The diffusion process inside the Gas Channel is captured by the species transport balance
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\boldsymbol{v}\,\nabla&space;Y_i=D_i\,\nabla^2&space;Y_i\end{align}">
</p>

where $Y_i$ denotes the mass fraction of species $i$.

### Catalyst Layer
At the Catalyst Layer the electrochemical reactions occur. At the anode, hydrogen undergoes the hydrogen oxidation
reaction (HOR), producing protons and electrons:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{H}_2\,\to\,2\mathrm{H}^&plus;&plus;2\mathrm{e}^-">
</p>

The proton-conductive Polymer Electrolyte Membrane allows for the diffusion of $\mathrm{H}^+$ from the Anode Catalyst
Layer to the Cathode Catalyst Layer.
The electrons are conducted by the porous layers and Bipolar Plate in the opposite direction of the protons from
Anode Catalyst Layer to Cathode Catalyst Layer.
The Oxygen Reduction Reaction on the cathode recombines the electrons and protons with oxygen to form water
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}1/2\mathrm{O}_2&plus;2\mathrm{H}^&plus;&plus;2\mathrm{e}^-\to\mathrm{H}_2\mathrm{O}">
</p>

The kinetics of the electrochemical reactions are modeled by the Butler-Volmer equation which relates the current
density $j$ with the overpotential $\eta$. For the cathode, the Butler-Volmer equation takes the following form:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}j=j_0\left(\frac{c_{\mathrm{O}_2}}{c_{{\mathrm{O}_2},\mathrm{ref}}}\right)^{\gamma_{\mathrm{O}_2}}\left[\exp{\left(\frac{\alpha_a\,F\,\eta}{R\,T}\right)}-\exp{\left(-\frac{\alpha_c\,F\,\eta}{R\,T}\right)}\right]\end{align}">
</p>

$j_0$ denotes the exchange current density, $T$ the temperature, $c_{O_2}$ the oxygen concentration on the 
Catalyst Layer, and $c_{O_2,ref} = 1$ kmol/m<sup>3</sup> its reference value. 
Note, that due to limited variance, the temperature is set to a constant value $T = 353.15$ K.
$F$ and $R$ are the Faraday constant and ideal gas constant, respectively.
$\gamma_{O_2}$ denotes the reaction order and $\alpha_a$ as well as $\alpha_c$ are modeling parameters to 
balance the backward and forward direction of the electrodes' reaction.
Here, only the cathode is considered, since it has the lower reaction rate and therefore limits the current density.

Since the Operating Point is not close to equilibrium, the second exponential term in Butler-Volmer equation
can be neglected, resulting in the Tafel equation:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}j=j_0\left(\frac{c_{\mathrm{O}_2}}{c_{{\mathrm{O}_2},\mathrm{ref}}}\right)^{\gamma_{\mathrm{O}_2}}\exp{\left(\frac{\alpha_a\,\mathrm{F}\,\eta}{R\,T}\right)}\end{align}">
</p>

The exchange current density $j_0$ is modeled by
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}j_0=j_0^\mathrm{ref}\,a_\mathrm{CL}\left(\frac{p_{\mathrm{O}_2}}{p_\mathrm{ref}}\right)^\gamma\exp{\left(-\frac{E}{R\,T}\left[1-\frac{T}{T_\mathrm{ref}}\right]\right)}\end{align}">
</p>

an Arrhenius equation with the activation energy $E = 72.4$ kJ/mol and reference temperature 
$T_{ref} = 298.15$ K.
The reference exchange current density is determined based on the methodology developed by Toussaint et al. to
$j_0^{ref} = 3.32$ A/cm<sup>2</sup>. However, to model the CFD data accurately with the PINN, the determined 
value was multiplied by a factor of 2.1 here.
Both the platinum loading $a_{CL}$ and the pressure dependence $\gamma$ equal unity.
The reference pressure is $p_{ref} = 101.3$ kPa. $p_{O_2}$, the oxygen partial pressure, 
can be determined based on the pressure $p$ and the molar fraction of oxygen $X_{O_2}$:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}p_{\mathrm{O}_2}=p\,X_{\mathrm{O}_2}">
</p>

To calculate the molar fraction of oxygen $X_{O_2}$ from its mass fraction $Y_{O_2}$, the molar mass
of oxygen $M_{O_2}$ and the other gaseous species are needed to compute the molar mass of the mixture $M$:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}X_{\mathrm{O}_2}&=\frac{M}{M_{\mathrm{O}_2}}\,Y_{\mathrm{O}_2}\\M&=\left(\frac{Y_{\mathrm{H}_2}}{M_{\mathrm{H}_2}}&plus;\frac{Y_{\mathrm{O}_2}}{M_{\mathrm{O}_2}}&plus;\frac{Y_{\mathrm{H}_2\mathrm{O}}}{M_{\mathrm{H}_2\mathrm{O}}}&plus;\frac{Y_{\mathrm{N}_2}}{M_{\mathrm{N}_2}}\right)^{-1}\end{align}">
</p>

Since the Tafel equation combines two unknowns, the definition of the overpotential $\eta$ is used
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\eta=-U_\mathrm{eq}-R_\mathrm{el}\,A_\mathrm{geom}\,j&space;">
</p>

to be able to find a solution for $\eta$ and $j$, given that $j_0$, $c_{O_2}$, and $U_{eq}$ are known.
Here, $U_{eq}$ denotes the equilibrium potential, $R_{el}$ the electrical resistance of the cell, 
and $A_{geom}$ the geometric area of the Membrane Electrode Assembly. The electrical resistance $R_{el}$ 
is not a single material property but results from the series connection of several partial resistances within the cell.
It is composed of the ohmic resistance of the Bipolar Plate and the porous layers, as well as the contact 
resistances at the interfaces between these components.

To model the electrical resistance, a fifth-order polynomial
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}R_{\mathrm{el}}(b,y)=a_5(b)\,y^5&plus;a_4(b)\,y^4&plus;a_3(b)\,y^3&plus;a_2(b)\,y^2&plus;a_1(b)\,y&plus;a_0(b)\end{align}">
</p>

in the cell width $y$ is used.
The coefficients of the polynomial $a_0, ..., a_5$ depend on the channel width $b$.

## Losses

### Gas Channel
To combine the ANN with these physical equations, a loss-term is formulated for each equation.
Since he inputs and outputs of the neural network are normalized according to the z-score
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\hat{\xi}=\frac{\xi-\mu_\xi}{\sigma_\xi}\\\hat{\zeta}=\frac{\zeta-\mu_\zeta}{\sigma_\zeta}\end{align}">
</p>

for an arbitrary input $\xi$ and arbitrary output $\zeta$. $\mu_\xi$ and $\mu_\zeta$ denote their mean. Their standard
deviation is given by $\sigma_\xi$ and $\sigma_\zeta$.
Therefore, the partial derivatives determined with Automatic Differentiation share the normalized scale of the PINN
outputs.
To retain the consistency of the physical equations, the partial derivatives need to be scaled.
Rewriting the equation of the z-score lead to
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\xi=\mu_\xi&plus;\sigma_\xi\,\hat{\xi}\\\zeta=\mu_\zeta&plus;\sigma_\zeta\,\hat{\zeta}\end{align}">
</p>

which results in the following scaling for the first partial derivatives:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\frac{\partial\zeta}{\partial\xi}=\frac{\partial\left(\mu_\zeta&plus;\sigma_\zeta\,\hat{\zeta}\right)}{\partial\left(\mu_\xi&plus;\sigma_\xi\,\hat{\xi}\right)}=\frac{\sigma_\zeta}{\sigma_\xi}\,\frac{\partial\hat{\zeta}}{\partial\hat{\xi}}\end{align}">
</p>
	
Similarly, the second partial derivatives are scaled:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\frac{\partial\zeta}{\partial\xi^2}=\frac{\sigma_\zeta}{\sigma_\xi^2}\,\frac{\partial\hat{\zeta}}{\partial\hat{\xi}^2}\end{align}">
</p>

Rewriting the continuity equation, momentum balance, and species transport, such that they each equate to zero and 
combining them with the scaled first and second derivatives results in the following loss terms for all 
collocation points $r \in \mathcal{R}$:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_\mathrm{Con}^r&=\frac{\sigma_u}{\sigma_x}\,\frac{\partial\hat{u}_r}{\partial\hat{x}_r}&plus;\frac{\sigma_w}{\sigma_z}\,\frac{\partial\hat{w}_r}{\partial\hat{z}_r}\\\mathcal{L}_{\mathrm{Mom},x}^r&=\frac{\sigma_x}{\sigma_p}\,\sigma_u\left[\frac{u_r}{\sigma_x}\,\frac{\partial\hat{u}_r}{\partial\hat{x}_r}&plus;\frac{w_r}{\sigma_z}\,\frac{\partial\hat{u}_r}{\partial\hat{z}_r}\right]-\frac{\sigma_x}{\sigma_p}\,\frac{\sigma_u}{\mathrm{Re}}\left[\frac{1}{\sigma_x^2}\,\frac{\partial^2\hat{u}_r}{\partial\hat{x}_r^2}&plus;\frac{1}{\sigma_z^2}\,\frac{\partial^2\hat{u}_r}{\partial\hat{z}_r^2}\right]&plus;\frac{\partial\hat{p}_r}{\partial\hat{x}_r}\\\mathcal{L}_{\mathrm{Mom},z}^r&=\frac{\sigma_z}{\sigma_p}\,\sigma_w\left[\frac{u_r}{\sigma_x}\,\frac{\partial\hat{w}_r}{\partial\hat{x}_r}&plus;\frac{w_r}{\sigma_z}\,\frac{\partial\hat{w}_r}{\partial\hat{z}_r}\right]-\frac{\sigma_z}{\sigma_p}\,\frac{\sigma_w}{\mathrm{Re}}\left[\frac{1}{\sigma_x^2}\,\frac{\partial^2\hat{w}_r}{\partial\hat{x}_r^2}&plus;\frac{1}{\sigma_z^2}\,\frac{\partial^2\hat{w}_r}{\partial\hat{z}_r^2}\right]&plus;\frac{\partial\hat{p}_r}{\partial\hat{z}_r}\\\mathcal{L}_\mathrm{Spe}^r&=\frac{u_r}{\sigma_x}\,\frac{\partial\hat{Y}_{i,r}}{\partial\hat{x}_r}&plus;\frac{w_r}{\sigma_z}\,\frac{\partial\hat{Y}_{i,r}}{\partial\hat{z}_r}-\frac{1}{\mathrm{Pe}_i}\left[\frac{1}{\sigma_x^2}\,\frac{\partial^2\hat{Y}_{i,r}}{\partial\hat{x}_r^2}&plus;\frac{1}{\sigma_z^2}\,\frac{\partial^2\hat{Y}_{i,r}}{\partial\hat{z}_r^2}\right]\end{align}">
</p>

Since the data set for the Gas Channel is constant in \(y\), its derivatives with respect to \(y\) are not shown here.
Note that momentum balances in $x$- and $z$-direction were each multiplied with the inverse scaling of the pressure 
gradient, i.e. $\sigma_x / \sigma_p$ and $\sigma_z / \sigma_p$, to reduce the order of magnitude of the loss.
Furthermore, the Reynolds-number
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{Re}=\bar{\rho}\,w_\mathrm{ref}L/\bar{\mu}">
</p>

and the Péclet-number
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{Pe}_i=w_\mathrm{ref}\,L/\bar{D}_i&space;">
</p>
 
were introduced to balance the scales of the convective and diffusive terms in the momentum balance and species 
transport.
$w_{ref} = 10$ m/s and $L = 0.05$ m denote the reference velocity and reference length in the main flow direction 
($z$-direction), respectively.
They both use average parameters
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\bar{\kappa}=\frac{1}{N_\mathcal{T}}\,\sum_{t=1}^{N_\mathcal{T}}\kappa_t&space;">
</p>

that are averaged over the $N_\mathcal{T}$ data points of the CFD data set ($\kappa = \rho, \mu, D_i$).

The boundary conditions are evaluated for each boundary point $s \in \mathcal{S}$ as follows:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_{\mathrm{BC},u}^s&=u^\mathrm{CFD}_s-u_s\\\mathcal{L}_{\mathrm{BC},w}^s&=w^\mathrm{CFD}_s-w_s\\\mathcal{L}_{\mathrm{BC},p}^s&=p^\mathrm{CFD}_s-p_s\\\mathcal{L}_{\mathrm{BC},Y_{\mathrm{O}_2}}^s&=Y^\mathrm{CFD}_{\mathrm{O}_2,s}-Y_{\mathrm{O}_2,s}\end{align}">
</p>

The physical losses and boundary losses for all of their respective data points are then squared and averaged using the
Mean Squared Error (MSE). For the physical losses subsets of the collocation points
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\mathcal{R}_{\mathrm{Batch},k}\subseteq\mathcal{R},\,k=1,\dots,n&space;">
</p>

with their respective number of data points
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}N_{\mathrm{Batch},k}=|\mathcal{R}_{\mathrm{Batch},k}|">
</p>

are used for the mini-batches during each epoch of the training with the Adam optimizer.
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_l=\frac{1}{N_{\mathrm{Batch},k}}\,\sum_{r=1}^{N_{\mathrm{Batch},k}}\left(\mathcal{L}_l^r\right)^2\\\text{with}\quad&space;r\in\mathcal{R}_{\mathrm{Batch},k},\,k=1,\dots,&space;n\end{align}">
</p>

Here, \(l\) denotes the different losses.
The optimization with L-BFGS uses the entire set of collocation data points, i.e. $\mathcal{R}_{Batch} = \mathcal{R}$.

During the entire training, the MSE of the boundary losses use all boundary points $N_\mathcal{S} = |\mathcal{S}|$.
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\mathcal{L}_{\mathrm{BC},l}=\frac{1}{N_\mathcal{S}}\,\sum_{s=1}^{N_\mathcal{S}}\left(\mathcal{L}_{\mathrm{BC},l}^s\right)^2&space;">
</p>

By training with the boundary loss during each epoch and iteration of Adam and L-BFGS, respectively, the PINN is able to 
learn the correct solution more easily, since fulfilling the boundaries is a vital part of learning said solution.

At last, the total loss is determined as a weighted sum of the physical losses and the boundary losses:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_\mathrm{Tot}=\mathcal{L}_\mathrm{Con}/\lambda_\mathrm{Con}&plus;\mathcal{L}_{\mathrm{Mom},x}/\lambda_{\mathrm{Mom},x}&plus;\mathcal{L}_{\mathrm{Mom},z}/\lambda_{\mathrm{Mom},z}&plus;\mathcal{L}_\mathrm{Spec}/\lambda_\mathrm{Spec}&plus;\mathcal{L}_{\mathrm{BC},u}/\lambda_{\mathrm{BC},u}&plus;\mathcal{L}_{\mathrm{BC},w}/\lambda_{\mathrm{BC},w}&plus;\mathcal{L}_{\mathrm{BC},p}/\lambda_{\mathrm{BC},p}&plus;\mathcal{L}_{\mathrm{BC},Y_{\mathrm{O}_2}}/\lambda_{\mathrm{BC},Y_{\mathrm{O}2}}\end{align}">
</p>

Here, weights $\lambda_l$ and $\lambda_{BC,l}$ are introduced to balance the contributions of the individual loss 
components. Their values were determined heuristically by trial and error and are:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\lambda_\mathrm{Con}=1\cdot&space;10^1,\qquad\lambda_{\mathrm{Mom},x}=1\cdot&space;10^{-2},\qquad\lambda_{\mathrm{Mom},z}=1\cdot&space;10^{-2},\qquad\lambda_\mathrm{Spec}=1\cdot&space;10^3\\\lambda_{\mathrm{BC},u}=2\cdot&space;10^{-2},\qquad\lambda_{\mathrm{BC},w}=2\cdot&space;10^{-1},\qquad\lambda_{\mathrm{BC},p}=1\cdot&space;10^4,\qquad\lambda_{\mathrm{BC},Y_{\mathrm{O}_2}}=1\cdot&space;10^{-5}\end{align}">
</p>

In PINN, individual loss components such as data, physical, and boundary losses can vary greatly in scale. Without 
proper weighting, some terms may dominate training, leading to an imbalance. Weighting factors $\omega$ balance these 
terms, promoting stable convergence and improved model accuracy.

### Catalyst Layer
The electrochemical reactions take place on the catalyst layer. Since only algebraic equations are used to model the 
electrochemical reactions, the entire CFD data set $\mathcal{T}$ is used for the training. Similar to the process in 
the Gas Channel, the Tafel equation and the surface overpotential equation are rewritten to evaluate to zero. The losses 
for all $t \in \mathcal{T}$ then read:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_\mathrm{Surf}^t&=\eta_t&plus;U_{\mathrm{eq},t}^\mathrm{CFD}&plus;R_{\mathrm{el},t}\,A_\mathrm{geom}\,j_t\\\mathcal{L}_\mathrm{Rea}^t&=\ln{\left(\frac{j_t}{j_{0,t}}\right)}-\gamma_{\ce{O2}}\,\ln{\left(\frac{c_{{\ce{O2}},t}^\mathrm{CFD}}{c_{{\ce{O2}},\mathrm{ref}}}\right)}-\frac{\alpha_a\,F\,\eta_t}{R\,T}\\\mathcal{L}_\mathrm{Width}^t&=b_t^\mathrm{CFD}-b_t\end{align}">
</p>

The loss containing the channel width \(b\) is included to enforce the learning of the width dependencies if more than 
one width is supplied during training.
Since only the algebraic equations are used to train the PINN to accurately model $j$ and $\eta$, the other quantities, 
such as the equilibrium voltage $U_{{eq},t}^{CFD}$, exchange current density $j_{0,t}$, and oxygen concentration 
$c_{O_2,t}^{CFD}$ are directly supplied by (superscript CFD) or calculated with CFD data.
The equations for the exchange current density, oxygen partial pressure, oxygen mole fraction, molar mass of the 
mixture, and electrical resistance are solved at each data point $t$ with the CFD data as well:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}j_{0,t}&=j_0^\mathrm{ref}\left(\frac{p_{\ce{O2},t}}{p_\mathrm{ref}}\right)^\gamma\exp{\left(-\frac{E}{R\,T}\left[1-\frac{T}{T_\mathrm{ref}}\right]\right)}\\p_{\ce{O2},t}&=p_t^\mathrm{CFD}\,X_{\ce{O2},t}\\X_{\ce{O2},t}&=\frac{M_t}{M_{\ce{O2}}}\,Y_{\ce{O2},t}^\mathrm{CFD}\\M_t&=\left(\frac{Y_{\ce{H2},t}^\mathrm{CFD}}{M_{\ce{H2}}}&plus;\frac{Y_{\ce{O2,t}}^\mathrm{CFD}}{M_{\ce{O2}}}&plus;\frac{Y_{\ce{H2O},t}^\mathrm{CFD}}{M_{\ce{H2O}}}&plus;\frac{Y_{\ce{N2},t}^\mathrm{CFD}}{M_{\ce{N2}}}\right)^{-1}\\R_{\mathrm{el},t}(b_t,y_t)&=a_5(b_t)\,y_t^5&plus;a_4(b_t)\,y_t^4&plus;a_3(b_t)\,y_t^3&plus;a_2(b_t)\,y_t^2&plus;a_1(b_t)\,y_t&plus;a_0(b_t)\end{align}">
</p>

Analogously to the Gas Channel, all losses are first squared and averaged using the MSE
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_l=\frac{1}{N_{\mathrm{Batch},k}}\,\sum_{t=1}^{N_{\mathrm{Batch},k}}\left(\mathcal{L}_l^t\right)^2\\\text{with}\quad&space;t\in\mathcal{T}_{\mathrm{Batch},k},\,k=1,\dots,m\end{align}">
</p>

with 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{T}_{\mathrm{Batch},k}\subseteq\mathcal{T},\,N_{\mathrm{Batch},k}=|\mathcal{T}_{\mathrm{Batch},k}|,\quad\text{for}\quad&space;k=1,\dots,m\end{align}">
</p>

The equality holds when using the L-BFGS optimizer $(k = 1$). Adam, again, works with mini-batches.
The total loss combines the individual MSEs of the loss terms into a weighted sum:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\mathcal{L}_\mathrm{Tot}=\mathcal{L}_\mathrm{Surf}/\lambda_\mathrm{Surf}&plus;\mathcal{L}_\mathrm{Rea}/\lambda_\mathrm{Rea}&plus;\mathcal{L}_\mathrm{Width}/\lambda_\mathrm{Width}\end{align}">
</p>
The weighting factors are as follows:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\color{white}\begin{align}\lambda_\mathrm{Surf}=1,\qquad\lambda_\mathrm{Rea}=1,\qquad\lambda_\mathrm{Width}=1\cdot&space;10^{-8}\end{align}">
</p>

## Loss implementation inside the code
The formulation of the losses can be found in `def _loss_function(...)` inside of [`training.training.py`](./training/training.py).
If `netwok = "ANN-CL"` or `network = "ANN-GC"` is selected in [`start/start.py`](./start/start.py), no physical losses are 
determined, and the network is solely trained based on data.
This is shown in the following code excerpt from `training.py`:
```python
if not self._is_pinn:
	self._update_outputs(inputs)
	# Only the data loss
	mse_data: tf.Tensor = tf.reduce_mean(
		tf.square(outputs - tf.stack(list(self._outs.values()), axis=1))
	)
	loss_comps["data"] = mse_data
	return mse_data, loss_comps
```
The function `_update_outputs(self, inputs: tf.Tensor) -> None` saves the predicted outputs of the neural network inside 
the dictionary `self._outs` which uses the strings of the variables("u", "v", "w", "p", and "Y_O2") as keys and returns 
the corresponding values as a `tf.Tensor`.
The difference between all of these predicted quantities and the values from the CFD (`outputs`) is squared and averaged 
to the single value `mse_data` in the fourth line of the code snippet above.
After this is done, `_loss_function(...)` already returns the loss `mse_data` and the dictionary `loss_comps` which 
contains the loss components retrievable by their name (e.g. "data" for the data loss here).
<br>
For the catalyst layer, `network = "PINN-CL"`, the physical losses can be described by algebraic equations. Thus, no 
gradients have to be determined. All physical losses can be found in the code block that starts with 
```python
if self._equations["react_rate"]:
	self._update_outputs(inputs)

	self._terms_and_scales.update({"AE.Current_Density": 1, "AE.Surface_Overpotential": 1, "CL.Width": 1e-8})
            
	phys_outs.update(["eta", "j"])
...
```
The code snippet above shows the same function to update the predicted outputs (`_update_outputs(...)` as in the 
previous code block.
The dictionary `self._terms_and_scales` is manually filled with entries of the scale used for the equations and their 
respective name. They are saved in the log files as well.
Similarly, the set phys_outs is updated with the output variables (here $j$ and $\eta$), such that the user can choose 
to print them to the console during training to verify the correctness of the learned model.
Importantly, the training requires the data set of the channel width (`widths`), since the model of the electrical 
resistance
`resist` (fifth-order polynomial) needs it to retrieve its coefficients.
Apart from the output variables that are learned by the neural network ("j" and "eta"), additional variables are needed 
for the physical equations.
All of these output variables have in common that their values are still normalized. To use them in said physical 
equations, they need to be denormalized.
This is done by using `_denormalize_var(self, var: str, use_vstats: bool) -> tf.Tensor` for the outputs to be learned 
and `__denormalize(self, vals: tf.Tensor, var: str, use_vstats: bool) -> tf.Tensor` for the additional variables. The 
latter needs the values `vals` that are used to denormalize the specified variable `var`. This function is therefore 
designed to denormalize the values that are directly supplied to the equations without being learned by the neural
network (additional variables).
<br>
However, if `network = "PINN-GC"` is selected in [`start/start.py`](./start/start.py), the gradients of the physical quantities are 
determined by using `_gradients(self, inputs: tf.Tensor) -> None`.
For that, all computations regarding the `inputs` and the predicted outputs (`_update_outputs(...)`) are "recorded" 
(similar to a "tape spool", hence the name `tf.GradientTape`). 
This subsequently allows to determine the derivatives of the predicted outputs with respected to the inputs using 
`tf.gradient()`.
This is done by the lambda functions `_first_deriv` and `_second_deriv` (`_sec_deriv_all`) for the first and second 
derivative, respectively. To explain these functions in a bit more detail, the important parts are discussed using 
`_first_deriv`, since they are the same for `_sec_deriv`:
```python
_first_deriv: Callable[[str, ], dict[str, tf.Tensor]] = lambda var: {
	# Dict comprehension -> Set gradient to zero if coordinate has no range (thickness of 2D plane)
	#                       or doesn't have a gradient (predictors)
	coord:  tape2.gradient(self._outs[var], inputs)[:, index]
	if index >= 0
	else tf.constant(0.0, dtype=tf.float32)
	for coord,index in self._in_indices.items()
}
```
They both accept a string `var` that represents the variable to determine the first or second derivatives for. 
`_first_deriv` and `_sec_deriv` return a dictionary that maps to the derivative with respect to the respective input 
("x", "y", and "z") by its name. Thus, `_firs_deriv["u"]["x"]` would be the first derivative of "u" with respect to "x".
The derivatives are determined for all inputs if the value range of the input inside the supplied data set is greater 
than zero. This means that unless the data set does not vary in that input (as is the case for "y" in the data set of 
the gas channel), the derivative with respect to that input will be determined. 
Conversely, if the data set is constant in an input, the value of the derivative with respect to it is set to zero.
Checking whether the data set does not vary in an input is done in `def get_in_indices(self) -> dict[str,int]` of 
[`data_handling/data.py`](./data_handling/data.py). There, a non-varying input gets an index of `-1` inside `self._in_indices`, a 
dictionary containing the input name ("x", "y", and "z") and their respective index (cf. last line of the code block 
above). 
This is the reason for the condition `if index >= 0` to determine whether the derivative of `var` is calculated with 
respect to the input or set to zero.
This use of these indices is in keeping track of the inputs position (or outputs positions, see `self._out_indices`) 
when supplying the data set to the neural network.
<br>
To go full circle, the use of derivatives is shown for the loss of the continuity equation in the code below:
```python
# Conti Equation
conti = tf.constant(0, dtype=tf.float32)
for coord, vel in {"x": "u", "y": "v", "z": "w"}.items():
	if self._in_indices[coord] == -1:
		continue

	conti += (
		self._get_var_std(vel, use_vstats) * self._first_derivs[vel][coord] /
		self._get_var_std(coord, use_vstats)
	)
```
Here the strings of the inputs and outputs are iterated over in the third line. The fourth line neglects inputs that 
were not considered for the derivative. Even though they were set to zero, they are excluded for numerical reasons.
Since the derivatives are determined for the normalized inputs and outputs, they need to be scaled by their standard 
deviation to retain the consistency of the physical equations.
For that, the function `_get_var_std(self, var: str, use_vstats: bool) -> tf.Tensor` returns the standard deviation of 
the input or output `var`. The option `use_vstats` 
toggles between the standard deviation of the normalization from the training (`use_vstats = False`) and the validation 
(`use_vstats = True`) data set.
<br>
In the loss of the momentum equation and species transport, the velocities are part of the convective term. Similarly 
to the derivatives, their values are normalized as well. For the equations they are needed with their physical scale. 
Similar to the equations on the catalyst layer, this can be done via `_denormalize_var(...)`.
<br>
Finally, there are code blocks that are enclosed by triplets of quotation marks. They include the losses of the conti, 
momentum, and species transport equation for the compressible case.
Even though they were not used for this publication, they can be used for further analyses. Their usage, however, 
require to remove the comment from the following line:
```python
#self._first_derivs["rho"]  = _first_deriv("rho")
```
Unsurprisingly, the data set used for training the neural network then needs to include the density for its boundary 
loss which in turn must be included in the training too.

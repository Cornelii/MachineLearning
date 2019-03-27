# Reinforcement Learning Basics

## I. Types of RL
1. Value Function-based Method

2. Policy Search Method

3. Actor-Critic Method (Policy Gradient)


## II. Markov Decision Process (MDP)
**Reinforcement Learning is method to solve MDP Problem**

MDP is composed of:
1. S: a finite sets of states
2. A: a finite sets of actions
3. Pr: state transition probability
4. R: rewards
* Discount Factor

## III. Reinforcement Learning

![RL](https://skymind.ai/images/wiki/simple_RL_schema.png)

#### 1.Policy
$$
    \begin{aligned}
    \pi(a|s) = P(a_t=a|s_t=s)
    \end{aligned}
    \\ Policy \space is \space action-selection \space at \space every \space states
$$

#### 2. Return
$$
    \begin{aligned}
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} +\dots
    \end{aligned}
    \\ R_i:random \space variables
    \\ G_i:also \space random \space variable \space combined
$$

#### 3. Value Function (state-value function)
$$
    \begin{aligned}
    V_\pi (s)=E_\pi [G_t|S_t=s]
    \end{aligned}
    \\ Expectation \space of \space Return \space on \space policy \space \pi
$$



**Bellman's Equation (Bellman Expectation Equation)**
$$
    \begin{aligned}
    \\V_\pi (s)&=E_\pi [G_t|S_t=s]
    \\V_\pi (s)&=E_\pi [R_{t+1} + \gamma (R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} +\dots )|S_t=s]
    \\V_\pi (s)&=E_\pi [R_{t+1}+ \gamma G_{t+1}|S_t=s]
    \\V_\pi (s)&=E_\pi [R_{t+1}] + E_\pi [ \gamma G_{t+1}|S_t=s]
    \\
    \\ E_\pi [ \gamma G_{t+1}|S_t=s] &= \gamma \Sigma_{a \in A_t}\pi(a|s) \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}V_\pi (s')
    \\ &= E_\pi [ \gamma V_\pi (S_{t+1})|S_t=s]
    \\
    \\V_\pi (s)&=\Sigma_{a \in A_t} \pi (a|s) R_{t+1}(s,a) + \gamma \Sigma_{a \in A_t}\pi(a|s) \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}V_\pi (s')
    \\
    \\V_\pi (s)&=\Sigma_{a \in A_t} \pi (a|s) [R_{t+1}(s,a) + \gamma  \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}V_\pi (s')]
    \\V_\pi (s)&=E_\pi [R_{t+1}+ \gamma V_\pi (S_{t+1})|S_t=s]
    \end{aligned}
$$


#### 4. Action Value Function (Q-function)

$$
\begin{aligned}
q_\pi(s,a) &= E_\pi [G_t|S_t=s, A_t=a]\\
V_\pi(s) &= \Sigma_{a \in A_t} \pi (a|s)q_\pi(s,a)\\
q_\pi(s,a) &= R_{t+1}(s,a) + \gamma  \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}V_\pi (s')\\
\end{aligned}
$$

**Optimal Value function & Policy**
$$
\begin{aligned}
V^*_\pi(s)= max_\pi(V_\pi(s))\\\\
\text{There exists a unique optimal value function (Bellman, 1957)}\\\\
\pi^*(s,a) = 
\begin{cases}
    1       & \quad \text{if } a=argmax_{a \in A} \space q^*(s,a)\\
    0  & \quad \text{otherwise}
\end{cases}
\end{aligned}
$$

**Bellman Optimal Equation about Q-function**
$$
\begin{aligned}
q^*_\pi(s,a) &= R_{t+1}(s,a) + \gamma  \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}V^*_\pi (s')\\
q^*_\pi(s,a) &= R_{t+1}(s,a) + \gamma  \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}\Sigma_{a' \in A_{t+1}}\pi^*(a'|s')q^*_\pi(s',a') \\
&= R_{t+1}(s,a) + \gamma  \Sigma_{s'\in S_{t+1}} P^{a}_{ss'}max_{a'}q^*_\pi(s',a') \\\\
\end{aligned}
\\\text{if }P^a_{ss'} = 1\\
q^*_\pi(s,a) = R_{t+1}(s,a) + \gamma  max_{a'}q^*_\pi(s',a') 
$$


#### 5. To solve RL probs => To solve Bellman Eqs


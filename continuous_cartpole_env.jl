# environment, julia port of the continuous cartpole environment to enable
# automatic differentiation through the environment


using PyCall
using LineaAlgebra
using OpenAIGym
using Random
using Distributions

# Key functions of cartpole environment (step, reset) are pure julia, so should be automatically
# differentiable

abstract type Env end

mutable struct ContinuousCartPole <: Env
    gravity::Float32
    masscart::Float32
    masspole::Float32
    total_mass::Float32
    length::Float32
    polemass_length::Float32
    force_mag::Float32
    tau::Float32
    min_action::Float32
    max_action::Float32
    theta_threshold::Float32
    x_threshold::Float32
    observation_space::Dict{}
    action_space::Dict{}
    state::Array{Float32,1}
    done::Boolean
    seed
end

function ContinuousCartPole(gravity = 9.8, masscart = 1.0, masspole = 0.1, length = 0.5, force_mag = 30.0, tau = 0.02,min_action=-1.0, max_action = 1.0, theta_thresh_param = 12.0, x_threshold=2.4,seed=nothing)
    total_mass = masspole + masscart
    polemass_length = masspole * length
    theta_threshold = (theta_thresh_param * 2 * π) /360.0
    action_space = Dict("low"=> min_action, "high"=> max_action)
    observation_space = Dict("low"=>theta_threshold * -2, "high"=>theta_threshold*2)
    s = seed(s)
    state = rand(eltype(env.state), Uniform(-0.05, 0.05), size(env.state))
    return ContinuousCartPole(gravity, masscart,masspole, total_mass,
                            length, polemass_length, force_mag, tau, min_action, max_action, theta_threshold
                            x_threshold, observation_space, action_space, state, false, s)
end

seed(s = nothing) = if !seed return Random.seed!(0) else return s end

function reset!(env::Env)
    env.done = false
    state = rand(eltype(env.state), Uniform(-0.05, 0.05), size(env.state))
    env.state = state
    return state
end

function stepPhysics(env, force)
    s = zeros(Float32,4)
    x, x_dot_theta, theta_dot = self.state
    costheta = cos(theta)
    sintheta = sin(theta)
    temp = (force + env.polemass_length * theta_dot^2 * sintheta) / env.total_mass
    thetaacc = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0/3.0 - env.masspole * costheta^2 / env.total_mass))
    xacc = temp - env.polemass_length * thetaaccc * costheta / env.total_mass
    s[1] = x + (env.tau * x_dot)
    s[2] = x_dot + (env.tau * xacc)
    s[3] = theta + (env.tau * theta_dot)
    s[4] = theta_dot + (env.tau * thetaacc)
    return s
end

function step(env, a)
    if a < env.action_space["low"] || a > env.action_space["high"]
        throw("Invalid action. Must be between $(env.action_space["low"]) and $(env.action_space["high"])")
    end
    a = convert(Float32, a) # just to make sure
    force = env.force_mag * a
    state = stepPhysics(force)
    x,x_dot, theta, theta_dot = self.state
    if env.done == true beyond = true else beyond = false
    if abs(x) > env.x_threshold || abs(theta) > env.theta_threshold
        env.done = true
    end
    if !env.done
        reward = 1.0
    end
    if beyond
        warn("This has already returned done == true. Any further steps are undefined behaviour"_)
        reward = 0.0
    end
    return state, reward
end

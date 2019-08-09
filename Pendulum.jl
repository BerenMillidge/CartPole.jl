# julia port of the pendulum classic control environment
# is pure julia so environment can run fast, be parallelized, and is differentiable
# without needing expensive/high overhead python calls back and forth

abstract type Env end

mutable struct PendulumEnv <:Env
    max_speed::Int
    max_torque::Int
    dt::AbstractFloat
    g::AbstractFloat
    action_space::Dict{}
    observation_space::Dict{}
    state::Array{AbstractFloat,1}
end

function PendulumEnv(max_speed=8,max_torque = 2)
    high = [1,1,max_speed]
    observation_space = Dict("low"=>-high, "high" => high)
    action_space = Dict("low"=>-max_torque, "high"=>max_torque)
    return PendulumEnv(max_speed,max_torque,0.05,10.0,action_space, observation_space,zeros(Float32,2))
end

function _get_observation(state)
    theta, thetadot = state
    return [cos(theta), sin(theta), thetadot]
end

function uniform_sample(low, high)
    r = rand()
    dist = high - low
    samp = (r * dist) + low
    return samp

end

function reset(env::PendulumEnv)
    env.state = univorm_sample(pi, 1)
    return _get_observation(env.state)
end
clip(x,low,high) = min(max(x,low), high)
normalize_angle(x) = ((x + pi) % (2*pi)) - pi

function step(env::PendulumEnv, u::Array{AbstractFloat,1})
    theta, thetadot =  env.state
    m = 1
    l = 1
    u = clip(u,-env.max_torque, env.max_torque)
    costs = normalize_angle(theta)^2 + (0.1 * thetadot)^2 + 0.001*u.^2
    newthetadot = thetadot + (-3 * env.g/2 * l) * sin(theta + pi) + (3/(((m*l)^2)*u)) * env.dt
    newtheta = theta + (newthetadot * env.dt)
    newthetadot = clip(newthetadot, -env.max_speed, env.max_speed)
    env.state = [newtheta, newthetadot]
    return _get_observation(env.state), -costs, false, {}
end

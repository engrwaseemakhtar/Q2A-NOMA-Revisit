% MATLAB code for Q-learning algorithm for sum rate optimization in Q²A-NOMA network

% Initialization
L = 10; % total number of devices (example value)
B = 20e6; % total bandwidth in Hz (example value)
P_noma = 10^(45/10) / 1000; % maximum allowable power in Watts (45dBm)
epsilon = 0.1; % exploration probability
num_iterations = 1000; % number of iterations
learning_rate = 0.1; % learning rate
discount_factor = 0.9; % discount factor

% Define state and action space
states = 1:L; % Define state space as device indices
actions = 0:P_noma/10:P_noma; % Define action space as power levels

% Initialize Q-table
Q = zeros(length(states), length(actions));

% Initial state and action
state = randi(length(states)); % Define initial state
action = randi(length(actions)); % Define initial action

% Initialize reward tracking
rewards = zeros(num_iterations, 1);

% Q-learning algorithm
for iter = 1:num_iterations
    total_reward = 0;
    for sub_band = 1:length(states)
        if rand() < epsilon
            % Exploration: select a random action
            action = randi(length(actions));
        else
            % Exploitation: select the action with max Q-value for the current state
            [~, action] = max(Q(state, :));
        end
        
        % Take action and observe new state and reward
        new_state = randi(length(states)); % Define state transition (example)
        reward = calculate_reward(new_state, action, B, P_noma); % Define reward calculation
        
        % Update Q-value
        Q(state, action) = Q(state, action) + learning_rate * ...
            (reward + discount_factor * max(Q(new_state, :)) - Q(state, action));
        
        % Accumulate reward
        total_reward = total_reward + reward;
        
        % Transition to the new state
        state = new_state;
    end
    rewards(iter) = total_reward;
end

% Extract optimal policy from Q-table
[~, optimal_policy] = max(Q, [], 2); % Define how to extract optimal policy

% Plot the reward
figure;
plot(rewards);
xlabel('Iteration');
ylabel('Total Reward');
title('Reward during Q-learning iterations');
grid on

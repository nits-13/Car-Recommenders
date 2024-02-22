from flask import Flask,render_template,request,redirect
import hashlib
from pymongo import MongoClient
import regex as re
from decision_tree import DecisionTree_CRS
from car_csp import CarRecommendationCSP
import heapq
import itertools

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")

db=client['CRS']
collections=db['Users']

def get_mongo_data():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Car"]
    collection = db["Car_Attributes"]
    mongo_data = list(collection.find({}))
    return mongo_data

#Decision Tree Algorithm Implementation
def get_recommendation_dt(user_input):
    mongo_data = get_mongo_data()
    car_system = DecisionTree_CRS(mongo_data)
    recommendation = car_system.get_recommendation(user_input)
    return recommendation

#CSP Algorithm Implementation
def get_recommendation_csp(user_input):
    mongo_data = get_mongo_data() 
    car_attributes = {}
    for document in mongo_data:
        document_without_id = {key: value for key, value in document.items() if key != '_id'}
        for key, value in document_without_id.items():
            if key not in car_attributes:
                car_attributes[key] = []
            car_attributes[key].append(value)
    
    car_variables = ['car_name', 'reviews_count', 'fuel_type', 'engine_displacement', 'no_cylinder', 'seating_capacity', 'transmission_type', 'fuel_tank_capacity', 'body_type', 'rating', 'starting_price', 'ending_price', 'max_torque_nm', 'max_torque_rpm', 'max_power_bhp', 'max_power_rp']
    user_constraints = {'reviews_count': user_input[0][0], 'fuel_type': user_input[0][1], 'engine_displacement': user_input[0][2],
                        'no_cylinder': user_input[0][3], 'seating_capacity': user_input[0][4], 'transmission_type': user_input[0][5],
                        'fuel_tank_capacity': user_input[0][6], 'body_type': user_input[0][7], 'rating': user_input[0][8], 'starting_price': user_input[0][9],
                        'ending_price': user_input[0][10], 'max_torque_nm': user_input[0][11], 'max_torque_rpm': user_input[0][12],
                        'max_power_bhp': user_input[0][13], 'max_power_rp': user_input[0][14]}

    car_constraints = {}
    for var, value in user_constraints.items():
        if var in car_variables:
            car_constraints[var] = [value]

    # Convert values in car_attributes to appropriate types (int, float) for consistency
    for var, values in car_attributes.items():
        if var in car_constraints and isinstance(values[0], (int, float)):
            car_constraints[var] = [int(value) if value.isdigit() else float(value) for value in car_constraints[var]]

    # Create the CSP object and solve
    car_csp = CarRecommendationCSP(
        variables=car_variables,
        domains=car_attributes,
        constraints=car_constraints
    )

    solutions = car_csp.solve()
    return solutions

    """
    if solutions:
        #print(f"Maximum Score: {max_score}")
        #print("Recommended Car Attributes:")
        for solution in solutions:
            print(solution)
    else:
        print("No solutions found for the given constraints.")
    """
#Heuristic Search Algorithm Implementation
# Function to calculate the score for a given car based on user constraints
def calculate_score(attributes, user_constraints):
    total_score = 0
    for feature, weightage in user_constraints.items():
        if feature in attributes and isinstance(attributes[feature], str):
            preferred_value, user_weightage = weightage
            total_score += int(preferred_value == attributes[feature]) * user_weightage
    return total_score

# Function to calculate the heuristic estimate for A* search
def heuristic_estimate(attributes, user_constraints):
    total_estimate = 0
    for feature, weightage in user_constraints.items():
        if feature in attributes and isinstance(attributes[feature], str):
            preferred_value, _ = weightage
            total_estimate += int(preferred_value != attributes[feature])  # Heuristic estimate
    return total_estimate

# A* search algorithm
def astar_search(attributes_list, user_constraints, max_score):
    open_set = []  
    closed_set = set()
    unique_id_counter = itertools.count()

    start_node = (0, next(unique_id_counter), None, attributes_list[0])

    heapq.heappush(open_set, start_node)

    while open_set:
        current_cost, _, parent, current_state = heapq.heappop(open_set)

        current_state_hashable = frozenset(current_state.items())

        if current_state_hashable in closed_set:
            continue

        closed_set.add(current_state_hashable)

        if parent is not None:
            score = calculate_score(current_state, user_constraints)
            if score > max_score[0]:
                max_score[0] = score
                recommended_car = current_state

        for neighbor_attributes in attributes_list:
            cost_to_neighbor = 1  
            heuristic_estimate_neighbor = heuristic_estimate(neighbor_attributes, user_constraints)
            total_cost_to_neighbor = current_cost + cost_to_neighbor + heuristic_estimate_neighbor

            heapq.heappush(open_set, (total_cost_to_neighbor, next(unique_id_counter), current_state, neighbor_attributes))

    return recommended_car, max_score[0]

def get_recommendation_hs(user_input):
    mongo_data = get_mongo_data()
    car_attributes_list = []
    for document in mongo_data:
        document_without_id = {key: value for key, value in document.items() if key != '_id'}
        car_attributes_list.append(document_without_id)

    # Specify user constraints with weightage
    user_constraints = {'reviews_count': [user_input[0][0], 5], 'fuel_type': [user_input[0][1], 8],
                        'engine_displacement': [user_input[0][2], 7], 'no_cylinder': [user_input[0][3], 6], 'seating_capacity': [user_input[0][4], 9],
                        'transmission_type': [user_input[0][5], 7], 'fuel_tank_capacity': [user_input[0][6], 6], 'body_type': [user_input[0][7], 8],
                        'rating': [user_input[0][8], 9], 'starting_price': [user_input[0][9], 5], 'ending_price': [user_input[0][10], 5],
                        'max_torque_nm': [user_input[0][11], 8], 'max_torque_rpm': [user_input[0][12], 6],
                        'max_power_bhp': [user_input[0][13], 7], 'max_power_rp': [user_input[0][14], 6]}

    # Initialize variables for recommended car
    max_score = [-1]
    recommended_car = None

    # Perform A* search
    recommended_car, max_score[0] = astar_search(car_attributes_list, user_constraints, max_score)
    return recommended_car

def username_regex(username):
    username_regex = "[a-zA-Z0-9_\-\.]\w{6,15}[\S]+"
    match_regex = re.search(username_regex, username)
    if match_regex:
        return True
    else:
        return False

def password_regex(password):
    password_regex = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@?#$%^&*+=]).{8}$"
    match_regex = re.search(password_regex, password)
    if match_regex:
        return True
    else:
        return False

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('homepage.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get the form data
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        if username_regex(username) and password_regex(password):
            if confirm_password == password:
                # Hash the username and password before storing
                hashed_username = hashlib.sha256(username.encode()).hexdigest()
                hashed_password = hashlib.sha256(password.encode()).hexdigest()

                # Store user data in MongoDB
                collections.insert_one({'username': hashed_username, 'password': hashed_password})

                return redirect('/input_rec')  # Redirect to the main page after signup
            else:
                return "Passwords don't match!"
        else:
            return "Invalid username or password format!"
    else:
        # Render the signup form template for GET request
        return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the form data
        entered_username = request.form['username']
        entered_password = request.form['password']

        hashed_entered_username = hashlib.sha256(entered_username.encode()).hexdigest()
        hashed_entered_password = hashlib.sha256(entered_password.encode()).hexdigest()

        # Retrieve user data from MongoDB
        user_data = collections.find_one({'username': hashed_entered_username, 'password': hashed_entered_password})

        if user_data:
            return redirect('/input_rec')
        elif collections.find_one({'username': hashed_entered_username}):
            return 'Incorrect password!'
        else:
            return "Looks like you're a new user, please signup first!"
    else:
        # Render the login form template for GET request
        return render_template('login.html')
    
@app.route('/input_rec', methods=['GET', 'POST'])
def input_preferences():
    if request.method == 'POST':
        reviews_count = request.form['reviews-count']
        fuel_type = request.form['fuel-type']
        engine_displacement = request.form['engine-displacement']
        number_of_cylinders = request.form['number-of-cylinders']
        seating_capacity = request.form['seating-capacity']
        transmission_type = request.form['transmission-type']
        fuel_tank_capacity = request.form['fuel-tank-capacity']
        rating = request.form['rating']
        starting_price = request.form['starting-price']
        ending_price = request.form['ending-price']
        max_torque_nm = request.form['max-torque-nm']
        max_torque_rpm = request.form['max-torque-rpm']
        max_power_bhp = request.form['max-power-bhp']
        max_power_rpm = request.form['max-power-rpm']
        body_type = request.form['body-type']

        user_input = [ 
            [reviews_count, fuel_type, engine_displacement, number_of_cylinders, seating_capacity, transmission_type, fuel_tank_capacity, body_type, rating, starting_price, ending_price, max_torque_nm, max_torque_rpm, max_power_bhp, max_power_rpm]
        ]
        
        """
        user_input1 = [ 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        """
        
        rec_list = []
        recommendation_dt = get_recommendation_dt(user_input)
        recommendation_csp = get_recommendation_csp(user_input)
        recommendation_hs = get_recommendation_hs(user_input)
        #print(recommendation_dt)
        #print(recommendation_csp)
        #print(recommendation_hs)
        rdt = ['Decision Tree', recommendation_dt]
        rcsp = ['CSP', recommendation_csp[-1][0]]
        rhs = ['Heuristic Search', recommendation_hs['car_name']]
        rec_list.append(rdt)
        rec_list.append(rcsp)
        rec_list.append(rhs)
        #print(rec_list)
        return render_template('view_cars.html', rec=rec_list)
    
    else:
        return render_template('input_rec.html')

if __name__ == '__main__':
    app.run(debug=True)

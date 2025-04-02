import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from flask import Flask, request, jsonify

app = Flask(__name__)

def main():
    # Load the dataset
    data = pd.read_csv('c:\\Users\\DELL\\Downloads\\archive (2)\\market.csv', sep=';')

    # Convert the dataset into a binary matrix (ensure all values are 0 or 1)
    data = data.astype(int)

    # Generate frequent itemsets using the Apriori algorithm
    frequent_itemsets = apriori(data, min_support=0.1, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    # Display the rules
    print("Generated Association Rules:")
    print(rules)

    # Save the rules to a CSV file
    output_path = 'c:\\Users\\DELL\\Downloads\\archive (2)\\association_rules_output.csv'
    rules.to_csv(output_path, index=False)
    print(f"Association rules saved to {output_path}")

@app.route('/rules', methods=['GET'])
def get_rules():
    try:
        # Load the rules from the CSV file
        rules_path = 'c:\\Users\\DELL\\Downloads\\archive (2)\\association_rules_output.csv'
        rules = pd.read_csv(rules_path)
        return jsonify(rules.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rules', methods=['POST'])
def generate_rules():
    try:
        # Load the dataset
        data = pd.read_csv('c:\\Users\\DELL\\Downloads\\archive (2)\\market.csv', sep=';')
        data = data.astype(int)

        # Generate frequent itemsets and rules
        frequent_itemsets = apriori(data, min_support=0.1, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        # Save the rules to a CSV file
        output_path = 'c:\\Users\\DELL\\Downloads\\archive (2)\\association_rules_output.csv'
        rules.to_csv(output_path, index=False)
        return jsonify({"message": "Rules generated and saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rules', methods=['PUT'])
def update_rules():
    try:
        # Accept new minimum support and lift threshold from the request
        min_support = float(request.json.get('min_support', 0.1))
        min_lift = float(request.json.get('min_lift', 1.0))

        # Load the dataset
        data = pd.read_csv('c:\\Users\\DELL\\Downloads\\archive (2)\\market.csv', sep=';')
        data = data.astype(int)

        # Generate frequent itemsets and rules with new thresholds
        frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

        # Save the updated rules to a CSV file
        output_path = 'c:\\Users\\DELL\\Downloads\\archive (2)\\association_rules_output.csv'
        rules.to_csv(output_path, index=False)
        return jsonify({"message": "Rules updated and saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
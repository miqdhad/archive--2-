import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

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

if __name__ == "__main__":
    main()
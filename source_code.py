#!/usr/bin/env python
# coding: utf-8

# ## Install mlxtend package

# In[1]:


# pip install mlxtend


# ## Import necessary libraries

# In[2]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import warnings
import time 


# ## Encoding data

# In[3]:


# One hot encoding for built-in apriori and fp-growth
def encode_data(df):
    # Extracting all unique items from Transaction column
    all_distinct_items = df['Transactions'].str.split(', ').explode().unique()
    encoded_transactions = []

    for transaction in df['Transactions']:
        transaction_items = transaction.split(', ')
        # Create a binary list for each transaction
        encoded_transactions.append([1 if item in transaction_items else 0 for item in all_distinct_items])
    # Returning the whole encoded dataset as dataframe     
    return pd.DataFrame(encoded_transactions, columns=all_distinct_items)


# ## Apriori Brute Force

# In[4]:


def itemset_frequencies(itemset, transactions):
    count = 0 # counter for number of transaction
    
    # Loop to check if the current item is a subset of the transaction, if yes then increment counter
    for transaction in transactions:
        if set(itemset).issubset(set(transaction)):
            count += 1
    return count

def apriori_bruteforce_with_rules(transactions, support_threshold, min_confidence):
     # Function to count the frequency of a given itemset in the transactions
    def itemset_frequencies(itemset, transactions):
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count

    # Function to find all frequent itemsets based on the support threshold
    def find_frequent_itemsets(transactions, support_threshold, distinct_items):
        frequent_itemsets = []
        current_itemsets = []
        
        print(">> Frequent 1-itemsets:\n")
        
        # Formatting the output
        max_itemset_length = max(len(str(set([item]))) for item in distinct_items) + 5
        print("-" * (max_itemset_length + 15))
        print(f"{'Itemset':<{max_itemset_length}} | {'Support':>8}")
        print("-" * (max_itemset_length + 15))

        # Loop to count the frequency of each individual item
        for item in distinct_items:
            count = itemset_frequencies([item], transactions)
            if count >= support_threshold:
                current_itemsets.append(frozenset([item]))  # Store 1-itemset if frequent
                frequent_itemsets.append((frozenset([item]), count))
                item_str1 = str(set([item]))
                print(f"{item_str1:<{max_itemset_length}} | {count:>8}")

        print("-" * (max_itemset_length + 15))

        k = 2
        while current_itemsets:
            next_itemsets = []
            
            # Generate new candidate itemsets by combining current itemsets
            for i, itemset1 in enumerate(current_itemsets):
                for itemset2 in current_itemsets[i + 1:]:
                    union_itemset = itemset1.union(itemset2)
                    if len(union_itemset) == k:
                        count = itemset_frequencies(union_itemset, transactions)
                        if count >= support_threshold and union_itemset not in next_itemsets:
                            next_itemsets.append(union_itemset)
                            frequent_itemsets.append((union_itemset, count))

            if next_itemsets:
                max_itemset_length = max(len(str(set(item))) for item in next_itemsets) + 5

                print(f"\n>> Frequent {k}-itemsets:\n")
                print("-" * (max_itemset_length + 15))
                print(f"{'Itemset':<{max_itemset_length}} | {'Support':>8}")
                print("-" * (max_itemset_length + 15))

                # Display the found frequent itemsets and their counts
                for itemset in next_itemsets:
                    count = itemset_frequencies(itemset, transactions)
                    itemset_str2 = str(set(itemset))
                    print(f"{itemset_str2:<{max_itemset_length}} | {count:>8}")

                print("-" * (max_itemset_length + 15))

            current_itemsets = next_itemsets
            k += 1

        # Displaying final frequent itemsets with their support
        print("\n>> Final list of frequent itemsets")
        max_itemset_length = max(len(str(set(itemset))) for itemset, _ in frequent_itemsets) + 5
        print("-" * (max_itemset_length + 20))
        print(f"{'Itemset':<{max_itemset_length}} | {'Support':>8}")  
        print("-" * (max_itemset_length + 20))

        total_transactions = len(transactions)
        for itemset, count in frequent_itemsets:
            support = (count / total_transactions) * 100
            itemset_str = str(set(itemset))
            print(f"{itemset_str:<{max_itemset_length}} | {support:>6.2f} %")

        print("-" * (max_itemset_length + 20))

        return frequent_itemsets

    # Function to generate association rules
    def generate_association_rules(frequent_itemsets, transactions, min_confidence):
        rules = []
        total_transactions = len(transactions)

        # Function to get all non-empty subsets of an itemset
        def get_subsets(itemset):
            subsets = []
            itemset_list = list(itemset)
            for i in range(1, 1 << len(itemset_list)):  # 1 << n gives 2^n combinations
                subset = [itemset_list[j] for j in range(len(itemset_list)) if (i & (1 << j))]
                if subset and len(subset) < len(itemset_list):
                    subsets.append(frozenset(subset))
            return subsets

        print("\n>> Association Rules:\n")
        rule_number = 1
        
        # Generating association rules for each frequent itemset
        for itemset, itemset_support_count in frequent_itemsets:
            subsets = get_subsets(itemset)
            for antecedent in subsets:
                consequent = itemset.difference(antecedent)
                if consequent:
                    antecedent_support_count = itemset_frequencies(antecedent, transactions)
                    confidence = itemset_support_count / antecedent_support_count if antecedent_support_count > 0 else 0

                    if confidence >= min_confidence:
                        support = (itemset_support_count / total_transactions) * 100
                        confidence_percent = confidence * 100

                        antecedent_str = ', '.join(antecedent)
                        consequent_str = ', '.join(consequent)
                        
                        print(f"Rule {rule_number}: {antecedent_str} --> {consequent_str} | support = {support:.2f} % | confidence = {confidence_percent:.2f} %")
                        
                        rules.append((antecedent_str, consequent_str, support, confidence_percent))
                        rule_number += 1

        # Handling the case where no rules are generated
        if not rules:
            print("No association rules were generated with the given support and confidence.")
            print("Please try again with different support and confidence values.")
        return rules

    distinct_items = list(set(item for transaction in transactions for item in transaction))
    
    frequent_itemsets = find_frequent_itemsets(transactions, support_threshold, distinct_items)
    generate_association_rules(frequent_itemsets, transactions, min_confidence)


# ## Apriori Builtin Function

# In[5]:


def apriori_builtin(df, min_support, min_confidence):
    
    df_encoded = encode_data(df)
    
    # Generate frequent itemsets using the Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=min_support/100, use_colnames=True)
    
    # Generate association rules from the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence/100)
    
    # Handling the case where no rules are generated
    if rules.empty:
        print(f"\nNo association rules were generated with minimum support = {min_support} % and minimum confidence = {min_confidence} %.")
        print("Please try again with different support and confidence values.")
    
    else:
        print("\nAssociation Rules:\n")
        for i, row in rules.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            support = row['support'] * 100
            confidence = row['confidence'] * 100
            print(f"Rule {i+1}: {antecedents} --> {consequents} | support = {support:.2f} % | confidence = {confidence:.2f} %")


# ## FP-growth

# In[6]:


def fp_growth(df, min_support, min_confidence):
   
    df_encoded = encode_data(df)
    
    # Generate frequent itemsets using the FP-Growth algorithm
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support/100, use_colnames=True)
    
    # Generate association rules from the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence/100)
    
    # Handling the case where no rules are generated
    if rules.empty:
        print(f"\nNo association rules were generated with minimum support = {min_support} % and minimum confidence = {min_confidence} %.")
        print("Please try again with different support and confidence values.")
    
    else:
        print("\nAssociation Rules:\n")
        for i, row in rules.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            support = row['support'] * 100
            confidence = row['confidence'] * 100
            print(f"Rule {i+1}: {antecedents} --> {consequents} | support = {support:.2f} % | confidence = {confidence:.2f} %")


# ## Main method

# In[9]:


def main(): 
    # Ignore warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Mapping of retail outlets to their corresponding CSV files
    csv_files = {
        1: '1_Amazon.csv',
        2: '2_BestBuy.csv',
        3: '3_KMart.csv',
        4: '4_Nike.csv',
        5: '5_Lidl.csv'
    }    
    
    user_choice = "Y"
    print("----------------------------------------------------------------------------------------------")
    print("                         Midterm Project - Association Rule Mining")
    print("----------------------------------------------------------------------------------------------")
    
    # Loop to allow the user to select datasets and run the algorithms until they choose to exit
    while user_choice.upper()!="N":
        try:
            print("\nChoose a retail outlet:")
            print("1.Amazon\n2.BestBuy\n3.KMart\n4.Nike\n5.Lidl\n6.To exit program")
            
            # Read user's choice
            user_input = int(input("\nEnter a number between 1 and 6: "))
            
            if user_input == 6: # To exit
                print("\nThank you for running the program! Goodbye!\n")
                break
                
            elif user_input in csv_files:
                # Reading the corresponding CSV file into a DataFrame
                df = pd.read_csv(csv_files[user_input])
                print(f"\nYou have selected the dataset - {csv_files[user_input]}\n")
                
                # Read minimum support and confidence from user 
                support = float(input("Enter the support value (1 to 100): "))
                confidence = float(input("Enter the confidence value (1 to 100): "))
                
                # Compute total transactions and support threshold
                total_transactions = len(df)
                support_threshold = (support / 100) * total_transactions

                # Extract distinct items from the transactions for frequency calculation
                all_items = df['Transactions'].str.split(', ').explode()
                distinct_items = all_items.unique()
                item_counts = all_items.value_counts()
                transactions = df['Transactions'].str.split(', ').tolist()
                
                # Frequent 1-itemsets
                frequent_items = item_counts[item_counts >= support_threshold]  
                
                # Calling the three functions and computing their execution time 
                
                # Apriori - Brute Force
                print("\n\n\t\t---------- Results of Apriori algorithm using brute force ----------\n")
                t1 = time.perf_counter()
                apriori_bruteforce_with_rules(transactions, support_threshold, confidence/100)
                apriori1_time = time.perf_counter() - t1
                               
                
                # Apriori - Builtin Function
                print("\n\n\t\t---------- Results of Apriori algorithm using builtin function ----------")
                t2 = time.perf_counter()
                apriori_builtin(df, support, confidence)
                apriori2_time = time.perf_counter() - t2
                
                
                # FP Tree
                print("\n\n\t\t--------------------- Results of FP Growth algorithm ---------------------")       
                t3 = time.perf_counter()
                fp_growth(df, support, confidence)
                fpg_time = time.perf_counter() - t3

                
                # Printing computational time
                print("\n\n\t\t-------------------------- Computation Time --------------------------\n")
                print("Brute force Apriori algorithm     = ", round(apriori1_time, 4), "s")
                print("Apriori using Built-in function   = ", round(apriori2_time, 4), "s")
                print("FP-Growth using Built-in function = ", round(fpg_time, 4), "s")
                
        except ValueError:
            print("Invalid input. Please try again.")
        
        user_choice = input("\nDo you want to continue (Enter Y/N): ")  
    print("\nThank you for running the program! Goodbye!\n")
    print("----------------------------------------------------------------------------------------------")


if __name__ == "__main__":  
    main()


# In[ ]:





# In[ ]:





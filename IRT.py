import pandas as pd
import numpy as np
from scipy.stats import norm


def calculate_eap_ability(responses, discrimination, difficulty, ability_range=(-4, 4), quadrature_points=41):
    """
    Calculate the Expected A Posteriori (EAP) ability estimate using the 2PL IRT model.

    Constants:
    ability_range: Tuple (min, max) specifying the range of ability values for integration
    quadrature_points: Number of quadrature points for numerical integration

    Returns:
        eap_ability: EAP ability estimate
        eap_standard_error: Standard error of the EAP ability estimate
    """
    # Define quadrature points and weights for numerical integration
    abilities = np.linspace(ability_range[0], ability_range[1], quadrature_points)
    weights = norm.pdf(abilities) / np.sum(norm.pdf(abilities))

    # Initialize arrays for ability computation
    posterior_numerator = np.zeros(quadrature_points)

    # Calculate the posterior probability
    for i in range(len(responses)):
        prob_correct = 1 / (1 + np.exp(-discrimination[i] * (abilities - difficulty[i])))
        prob_incorrect = 1 - prob_correct

        # Update posterior probabilities
        posterior_numerator += (responses[i] * np.log(prob_correct) + (1 - responses[i]) * np.log(prob_incorrect))

    posterior_denominator = np.exp(posterior_numerator - np.max(posterior_numerator))  # Avoid overflow
    posterior_denominator /= posterior_denominator.sum()

    # Compute ability
    eap_ability = np.sum(abilities * posterior_denominator)

    # Compute the standard error of the ability estimate
    eap_variance = np.sum((abilities - eap_ability) ** 2 * posterior_denominator)
    eap_standard_error = np.sqrt(eap_variance)

    return eap_ability, eap_standard_error




# Function 2: Calculate Fisher information for a given ability level and parameters
# of item difficulty and item discrimination. Resulting Fisher information values are
# a float type numbers where the larger the value, the more informative a question.
def calculate_fisher_information(data, item_id, ability):
    # Filter the dataset for the specified item_id
    item_data = data[data['item_id'] == item_id]

    if item_data.empty:
        raise ValueError(f"Item ID {item_id} not found in the dataset.")

    # Extract difficulty and discrimination values for the item
    difficulty = item_data['difficulty'].values[0]
    discrimination = item_data['discrimination'].values[0]

    # Compute the probability of a correct response under the 2PL model
    prob_correct = 1 / (1 + np.exp(-discrimination * (ability - difficulty)))
    prob_incorrect = 1 - prob_correct

    # Fisher information for the specified item
    fisher_info = discrimination ** 2 * prob_correct * prob_incorrect

    return fisher_info

if __name__ == "__main__":
    # Example usage:

    # Load the sample dataset
    data = pd.read_csv('data/sample_data.csv')

    # Select three questions to estimate ability (I randomly selected three questions
    # for this example)
    # selected_items = data[data['item_id'].isin([2053236, 2053237, 2053238])]
    selected_items = data

    # Calculate EAP ability for a student
    responses = selected_items['response'].values
    discrimination = selected_items['discrimination'].values
    difficulty = selected_items['difficulty'].values

    example_eap_ability = calculate_eap_ability(responses, discrimination, difficulty)
    print(f"EAP Ability Estimate: {example_eap_ability}")

    # Example usage:

    # Load the sample dataset
    data = pd.read_csv('data/sample_data.csv')

    # Calculate Fisher information for an example item and ability
    example_fisher_info = calculate_fisher_information(data, item_id = 2053236, ability = -1)
    print(f"Fisher Information: {example_fisher_info}")

def batch_fisher_information(difficulty, discrimination, ability):
    """
    Calculate the Fisher information for multiple items using the 2PL model.

    Parameters:
        difficulty (array-like): Item difficulties. (n_items,)
        discrimination (array-like): Item discriminations. (n_items,)
        theta (float): Ability level.

    Returns:
        numpy.ndarray: Fisher information values for each item. Shape: (n_items,)
    """
    # Compute the probability of a correct response under the 2PL model
    prob_correct = 1 / (1 + np.exp(-discrimination * (ability - difficulty)))
    prob_incorrect = 1 - prob_correct

    # Fisher information 
    fisher_info = discrimination ** 2 * prob_correct * prob_incorrect

    return fisher_info
    
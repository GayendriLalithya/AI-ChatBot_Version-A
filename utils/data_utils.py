import json
import re


def load_knowledge_base(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def format_product_details(item):
    description = item.get('description', {})
    formatted_details = f"Brand: {item['brand']}\n"
    for key, value in description.items():
        formatted_key = key.replace('_', ' ').title()
        formatted_details += f"{formatted_key}: {value}\n"
    return formatted_details.strip()


def find_product_details(product_name, knowledge_base):
    product_name = product_name.lower()
    print(f"Searching for product: {product_name}")
    for item in knowledge_base:
        name = item['name'].lower()
        brand = item['brand'].lower()
        print(f"Checking against: {name} by {brand}")
        # Use regular expression to match all words in the product_name
        if re.search(r'\b' + re.escape(product_name) + r'\b', name) or re.search(r'\b' + re.escape(product_name) + r'\b', brand):
            print(f"Found match: {name} by {brand}")
            return format_product_details(item)
    return "Sorry, I couldn't find any details for that product."

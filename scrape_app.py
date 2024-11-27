import json, os, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import requests
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import Counter

load_dotenv()


def plot_category_pie_chart(results: List[Dict]):
    """
    Generates a pie chart showing the distribution of categories.

    Args:
        results (list): A list of dictionaries containing review classifications and sentiments.
    """
    # Count categories
    category_counts = Counter()
    for item in results:
        for category in item['categories'].get('categories', []):
            category_counts[category] += 1

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.pie(
        category_counts.values(),
        labels=category_counts.keys(),
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab10.colors,
    )
    plt.title("Distribution of Categories", fontsize=16)
    plt.axis('equal')  # Equal aspect ratio to ensure pie is circular.
    plt.show()


def plot_subcategory_pie_charts(results: List[Dict]):
    """
    Generates individual pie charts for subcategories of each category.

    Args:
        results (list): A list of dictionaries containing review classifications and sentiments.
    """
    # Collect subcategories for each category
    subcategory_data = {}
    for item in results:
        for category in item['categories'].get('categories', []):
            subcategories = item['categories'].get(category, [])
            if category not in subcategory_data:
                subcategory_data[category] = Counter()
            subcategory_data[category].update(subcategories)

    # Plot individual pie charts for each category
    for category, subcategories in subcategory_data.items():
        plt.figure(figsize=(8, 8))
        plt.pie(
            subcategories.values(),
            labels=subcategories.keys(),
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.Set3.colors,
        )
        plt.title(f"Subcategories for {category}", fontsize=16)
        plt.axis('equal')  # Equal aspect ratio to ensure pie is circular.
        plt.show()

def plot_category_chart(results: List[Dict]):
    """
    Generates a bar chart showing the distribution of categories.

    Args:
        results (list): A list of dictionaries containing review classifications and sentiments.
    """
    # Count categories
    category_counts = Counter()
    for item in results:
        for category in item['categories'].get('categories', []):
            category_counts[category] += 1

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(category_counts.keys(), category_counts.values(), color='skyblue')
    plt.title("Distribution of Categories", fontsize=16)
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_subcategory_charts(results: List[Dict]):
    """
    Generates individual bar charts for subcategories of each category.

    Args:
        results (list): A list of dictionaries containing review classifications and sentiments.
    """
    # Collect subcategories for each category
    subcategory_data = {}
    for item in results:
        for category in item['categories'].get('categories', []):
            subcategories = item['categories'].get(category, [])
            if category not in subcategory_data:
                subcategory_data[category] = Counter()
            subcategory_data[category].update(subcategories)

    # Plot individual charts for each category
    for category, subcategories in subcategory_data.items():
        plt.figure(figsize=(10, 6))
        plt.bar(subcategories.keys(), subcategories.values(), color='lightgreen')
        plt.title(f"Subcategories for {category}", fontsize=16)
        plt.xlabel("Subcategories", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

class LlamaReviewClassifier:
    def __init__(self, groq_api_key: str):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }
        self.llama_model = "llama-3.2-90b-text-preview"
    
    def classify_review_with_llama(self, review: str, taxonomy: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Classify a review into categories using Llama.

        Args:
            review (str): The review text.
            taxonomy (dict): Predefined categories and their keywords.

        Returns:
            dict: Dictionary containing the matched categories and sentiment.
        """
        prompt = f"""
        You are an intelligent assistant. Classify the following review into predefined categories based on the taxonomy provided. 
        Include only the categories and the sub-categories that match.
        
        Review: "{review}"
        
        Taxonomy: {json.dumps(taxonomy, indent=2)}
        
        Output the results as a JSON in the format:
        {{
            "categories": [<matched categories>],
            "<matched_category_1>": [<matched sub_categories>],
            "<matched_category_2>": [<matched sub_categories>],
            "<matched_category_3>": [<matched sub_categories>]
            ... and so on
        }}

        If no category matches, then return {{"categories" : []}}
        """
        
        #print(f"prompt is {prompt}")
        payload = {
            "model": self.llama_model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1000,
        }
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        response_data = response.json()
        
        try:
            result = response_data["choices"][0]["message"]["content"]
            print(f"Raw response is {result}")
            result_json = json.loads(result)
            return result_json
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"categories": []}

def analyze_reviews(reviews, taxonomy):
    """
    Analyzes a list of reviews for categories and sentiment.

    Args:
        reviews (list): A list of review texts.
        taxonomy (dict): A dictionary of categories and keywords.

    Returns:
        list: A list of dictionaries containing review classifications and sentiments.
    """

    review_classifier = LlamaReviewClassifier(groq_api_key = os.getenv('GROQ_API_KEY'))
    results = []
    for review in reviews:
        print(f"Analyzing review : {review}")
        categories = review_classifier.classify_review_with_llama(review, taxonomy)
        results.append({
            "review": review,
            "categories": categories,
        })
    return results

def display_results(results):
    """
    Displays the results of the review analysis.

    Args:
        results (list): A list of dictionaries containing review classifications and sentiments.
    """
    for item in results:
        print(f"Review: {item['review']}")
        print(f"Categories: {item['categories']}")
        print("-" * 50)

# Function to extract text from the specified element on the page
def extract_text_from_all_elements(url, parent_xpath, child_xpath_template):
  """
  Extract text content from elements within the page at the specified URL.
  Args:
      url (str): The URL of the webpage.
  Returns:
      str: Concatenated text content from the page, or None if an error occurs.
  """
  # Set up the Chrome WebDriver
  options = Options()
  options.add_argument('--headless')
  options.add_argument('--no-sandbox')
  options.add_argument('--disable-dev-shm-usage')
  options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
  driver = webdriver.Chrome(options=options)
  
  try:
      # Navigate to the specified URL
      driver.get(url)
      # Wait for the page to load (adjust the timeout as needed)
      wait = WebDriverWait(driver, 10)
      # Get the page source
      parent_element = wait.until(EC.presence_of_element_located((By.XPATH, parent_xpath)))

      # Find the element with ID 'cmsContent'
      # Count the number of children
      children = parent_element.find_elements(By.XPATH, "./*")  # "./*" selects all direct children
      child_count = len(children)
      print(f"Number of children: {child_count}")

      # Extract text from each child and build child-specific XPath
      child_texts = {}
      for index in range(child_count):
          child_xpath = child_xpath_template.format(index=index + 1)
          try:
              child_element = driver.find_element(By.XPATH, child_xpath)
              child_texts[child_xpath] = child_element.text.strip() if child_element.text else "No text found"
          except Exception as e:
              child_texts[child_xpath] = f"Error: {e}"
      return {"child_count": child_count, "child_texts": child_texts}
      
  except Exception as e:
      print(f"An error occurred while processing {url}: {e}")
      return None
  finally:
      # Ensure the WebDriver is closed
      driver.quit()

if __name__ == "__main__":
    # Input details
    urls = ["https://www.amazon.in/Fargo-Womens-Stylish-Handbag-Shoulder/dp/B0D4J5NGFN/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_1?pd_rd_w=PvGvw&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=BB31HRK3ZYPXFBV4790X&pd_rd_wg=8JPLm&pd_rd_r=44d697ea-2529-4f31-a292-667582b6b35f&pd_rd_i=B0D4J5NGFN&th=1", "https://www.amazon.in/Fargo-Handcrafted-Handbag-College-Grey_FGO-525/dp/B0BV6VSH3Q?ref_=ast_sto_dp", "https://www.amazon.in/dp/B0BV6TFRCB/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_0?psc=1&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=S0HK3X37X1CNX0GRY843&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&", "https://www.amazon.in/Fargo-Stylish-Textured-Sling-Ladies/dp/B0DFQHSNF7/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_3?pd_rd_w=41fYp&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=S0HK3X37X1CNX0GRY843&pd_rd_wg=H7SI6&pd_rd_r=caf14b3f-b0ea-4074-8fa8-47a97d3cc793&pd_rd_i=B0DFQHSNF7", "https://www.amazon.in/dp/B0D4MKL3WY/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_1?psc=1&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=S0HK3X37X1CNX0GRY843&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&", "https://www.amazon.in/Fargo-Stylish-Textured-Sling-Ladies/dp/B0DFQHSNF7/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_1?pd_rd_w=RAkmI&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=WM6PZXNSY1HT9YW18SDP&pd_rd_wg=2JLxg&pd_rd_r=115d431c-0080-4579-873c-a1ca0800ff9a&pd_rd_i=B0DFQHSNF7", "https://www.amazon.in/Fargo-Leatherette-Shoulder-Womens-Yellow_FGO-638/dp/B0CBCPK1CX/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_2?pd_rd_w=RAkmI&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=WM6PZXNSY1HT9YW18SDP&pd_rd_wg=2JLxg&pd_rd_r=115d431c-0080-4579-873c-a1ca0800ff9a&pd_rd_i=B0CBCPK1CX"]
    parent_xpath = "/html/body/div[2]/div/div[5]/div[27]/div/div/div/div/div[2]/div/div[2]/span[1]/div/div/div[3]/div[3]/div"
    child_xpath_template = "/html/body/div[2]/div/div[5]/div[27]/div/div/div/div/div[2]/div/div[2]/span[1]/div/div/div[3]/div[3]/div/div[{index}]/div/div/div[4]/span/div/div[1]/span"

    # Call the function
    final_result = []
    c = 0
    for url in urls:
      results = extract_text_from_all_elements(url, parent_xpath, child_xpath_template)
      final_result.append(results)

      reviews = []
      for child_xpath, text in results['child_texts'].items():
        reviews.append(text)

      if c > 4:
        break
      else:
        c += 1
        time.sleep(10)
  
  
      # Predefined taxonomy with keywords
    taxonomy = {
        "Design and Style": ["trendy", "classic", "color", "style", "aesthetic"],
        "Material and Craftsmanship": ["durable", "leather", "fabric", "stitching", "quality", "material"],
        "Functionality": ["storage", "pockets", "lightweight", "comfortable", "versatile"],
        "Pricing": ["affordable", "expensive", "value for money"],
        "Brand and Packaging": ["brand reputation", "packaging", "unboxing", "dust bag"],
        "Delivery and Condition": ["delivery time", "damaged", "packaging quality"],
        "Customer Support": ["responsive", "return", "exchange", "warranty"]
    }

    # Analyze reviews
    review_analysis = analyze_reviews(reviews, taxonomy)

    # Display results
    display_results(review_analysis)

    # Plot overall category chart
    plot_category_pie_chart(review_analysis)

    # Plot subcategory charts
    plot_subcategory_pie_charts(review_analysis)

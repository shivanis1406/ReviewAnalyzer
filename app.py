import json, os, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
import requests
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import csv

load_dotenv()



def plot_category_pie_chart(results: List[Dict], save_path: str = None):
    """
    Generates separate pie charts for positive and negative category distributions.
    
    Args:
        results (list): A list of dictionaries containing review classifications.
    """

    # Separate positive and negative categories
    sentiments = ['Positive', 'Negative']
    category_counts = {'Positive': Counter(), 'Negative': Counter()}
    
    for raw_item in results:
        item = raw_item["response"]
        # Extract labels and count categories based on sentiment
        for label in item.get('labels', []):
            sentiment = label['sentiment']
            category_counts[sentiment].update([label['category']])
        
    # Plot pie charts for each sentiment
    for sentiment in sentiments:
        category_count = category_counts[sentiment]
        # Sort categories by count
        sorted_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)
        categories = [cat for cat, count in sorted_categories]
        counts = [count for cat, count in sorted_categories]
        
        if not categories:
            continue
        
        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.pie(
            counts,
            labels=categories,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            pctdistance=0.85
        )
        
        # Add total review count as a title
        plt.title(f"{sentiment} Category Distribution)", fontsize=16)
        
        # Ensure pie is circular
        plt.axis('equal')
        
        # Add a legend for better readability
        plt.legend(categories, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Adjust layout to prevent cutting off the legend
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_{sentiment}_subcategory_distribution.png")
        else:
            plt.show()
        
        plt.close()


def plot_subcategory_pie_charts(results: List[Dict], save_path: str = None, N: int = 100):
    """
    Generates separate pie charts for positive and negative subcategories of each category.
    
    Args:
        results (list): A list of dictionaries containing review classifications.
    """
    # Collect subcategory data for each sentiment
    sentiments = ['Positive', 'Negative']
    subcategory_data = {'Positive': {}, 'Negative': {}}
    
    for raw_item in results:
        item = raw_item["response"]
        for label in item.get('labels', []):
            sentiment = label['sentiment']
            category = label['category']
            subcategory = label['sub_category']
            
            if category not in subcategory_data[sentiment]:
                subcategory_data[sentiment][category] = Counter()
            
            subcategory_data[sentiment][category].update([subcategory])
    
    # Plot pie charts for each sentiment and category
    for sentiment in sentiments:
        category_data = subcategory_data[sentiment]
        
        # Calculate number of subplots needed
        num_categories = len(category_data)
        
        if num_categories == 0:
            continue  # Skip if no data for this sentiment
        
        # Create subplots for each category
        cols = min(2, num_categories)  # Max 2 columns
        rows = (num_categories + 1) // 2  # Calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle(f"{sentiment} Subcategory Distribution by Category", fontsize=16)
        
        axes = axes.flatten() if num_categories > 1 else [axes]
        
        for idx, (category, subcategories) in enumerate(category_data.items()):
            sorted_subcategories = sorted(subcategories.items(), key=lambda x: x[1], reverse=True)
            sub_labels = [sub for sub, count in sorted_subcategories]
            sub_counts = [count for sub, count in sorted_subcategories]
            
            # Create color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(sub_labels)))
            
            axes[idx].pie(
                sub_counts,
                labels=sub_labels,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                pctdistance=0.85
            )
            
            axes[idx].set_title(f"{category} ({sentiment})", fontsize=12)
        
        # Remove unused subplots
        for idx in range(num_categories, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_{sentiment}_subcategory_distribution.png")
        else:
            plt.show()
        plt.close()

def plot_reason_pie_charts(results: List[Dict], save_path: str = None):
    """
    Generates pie charts for the reasons per sub-category.

    Args:
        results (list): A list of dictionaries containing review classifications.
    """
    # Collect reason data for each sub-category and sentiment
    sentiments = ['Positive', 'Negative']
    reason_data = {'Positive': {}, 'Negative': {}}
    
    for raw_item in results:
        item = raw_item['response']
        for label in item.get('labels', []):
            sentiment = label['sentiment']
            subcategory = label['sub_category']
            reason = label['reason']
            
            if subcategory not in reason_data[sentiment]:
                reason_data[sentiment][subcategory] = Counter()
            
            reason_data[sentiment][subcategory].update([reason])
    
    # Plot pie charts for reasons within each sub-category
    for sentiment in sentiments:
        subcategory_data = reason_data[sentiment]
        
        # Create subplots for each subcategory
        num_subcategories = len(subcategory_data)
        
        if num_subcategories == 0:
            continue  # Skip if no data for this sentiment
        
        cols = min(2, num_subcategories)  # Max 2 columns
        rows = (num_subcategories + 1) // 2  # Calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle(f"{sentiment} Reasons Distribution by Sub-category", fontsize=16)
        
        axes = axes.flatten() if num_subcategories > 1 else [axes]
        
        for idx, (subcategory, reasons) in enumerate(subcategory_data.items()):
            sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
            reason_labels = [reason for reason, count in sorted_reasons]
            reason_counts = [count for reason, count in sorted_reasons]
            
            # Create color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(reason_labels)))
            
            axes[idx].pie(
                reason_counts,
                labels=reason_labels,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                pctdistance=0.85
            )
            
            axes[idx].set_title(f"{subcategory} ({sentiment})", fontsize=12)
        
        # Remove unused subplots
        for idx in range(num_subcategories, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_{sentiment}_subcategory_reason_distribution.png")
        else:
            plt.show()

        plt.close()


class LlamaReviewClassifier:
    def __init__(self, groq_api_key: str):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }
        self.llama_model = "llama-3.2-90b-vision-preview"
    
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
        You are an intelligent assistant. Classify the following review into predefined categories and sub-categories based on the provided taxonomy. 
        Match the sentiment (positive/negative) for each sub-category where applicable. 

        Review: "{review}"

        Taxonomy: {json.dumps(taxonomy, indent=2)}

        Output result as only a JSON based on the above Taxonomy. Following is an example JSON

        {{
        "labels": [
            {{
            "category": "Design and Style",
            "sub_category": "Trendy",
            "sentiment": "Positive",
            "reason": "stylish"
            }},
            {{
            "category": "Material and Craftsmanship",
            "sub_category": "Durable",
            "sentiment": "Negative",
            "reason": "fragile"
            }},
            {{
            "category": "Pricing",
            "sub_category": "Affordable",
            "sentiment": "Positive",
            "reason": "great value"
            }}
        ]
        }}

        if no suitable label is found, return {{"labels" : []}}
        Sentiment can either be positive or negative. It can't be neutral
        Output must only be a JSON and nothing else
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
            print(f"Response is {response_data}")
            result = response_data["choices"][0]["message"]["content"]
            result_json = json.loads(result)
            print(f"Response is {result_json}")
            return result_json
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"labels": []}

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
        time.sleep(20)
        categories = review_classifier.classify_review_with_llama(review, taxonomy)
        results.append({
            "review": review,
            "response": categories,
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
        print(f"Response: {item['response']}")
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
  options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  #driver = webdriver.Chrome(options=options)
  driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

  try:
      # Navigate to the specified URL
      driver.get(url)

      # Add cookies after loading the initial page
      #driver.add_cookie({'name': "JSESSIONID", 'value': '9397710162CD6095B337FBE87A0F53BC'})

      # Refresh the page to apply cookies
      #driver.refresh()

      # Wait for the page to load (adjust the timeout as needed)
      wait = WebDriverWait(driver, 30)
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

  except WebDriverException as e:
      print(f"WebDriver error: {e}")
      return None
  except TimeoutException as e:
      print(f"Timeout error: {e}")
      return None
  except Exception as e:
      print(f"An error occurred while processing {url}: {e}")
      return None
  finally:
      # Ensure the WebDriver is closed
      driver.quit()

def get_reviews(file_path):
    """
    Reads the last column from a CSV file and appends the values to the reviews list.

    Args:
        file_path (str): Path to the CSV file.
        reviews (list): List to store the collected reviews.

    Returns:
        list: Updated list with reviews from the CSV file.
    """
    reviews = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Skip the header row
            last_column_index = len(headers) - 1  # Index of the last column
            
            # Extract the last column from each row and append to reviews
            for row in reader:
                if len(row) > last_column_index:  # Ensure the row has enough columns
                    reviews.append(row[last_column_index].strip())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return reviews

if __name__ == "__main__":
    # Input details
    #urls = ["https://www.amazon.in/Fargo-Womens-Stylish-Handbag-Shoulder/dp/B0D4J5NGFN/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_1?pd_rd_w=PvGvw&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=BB31HRK3ZYPXFBV4790X&pd_rd_wg=8JPLm&pd_rd_r=44d697ea-2529-4f31-a292-667582b6b35f&pd_rd_i=B0D4J5NGFN&th=1", "https://www.amazon.in/Fargo-Handcrafted-Handbag-College-Grey_FGO-525/dp/B0BV6VSH3Q?ref_=ast_sto_dp", "https://www.amazon.in/dp/B0BV6TFRCB/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_0?psc=1&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=S0HK3X37X1CNX0GRY843&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&", "https://www.amazon.in/Fargo-Stylish-Textured-Sling-Ladies/dp/B0DFQHSNF7/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_3?pd_rd_w=41fYp&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=S0HK3X37X1CNX0GRY843&pd_rd_wg=H7SI6&pd_rd_r=caf14b3f-b0ea-4074-8fa8-47a97d3cc793&pd_rd_i=B0DFQHSNF7", "https://www.amazon.in/dp/B0D4MKL3WY/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_1?psc=1&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=S0HK3X37X1CNX0GRY843&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&", "https://www.amazon.in/Fargo-Stylish-Textured-Sling-Ladies/dp/B0DFQHSNF7/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_1?pd_rd_w=RAkmI&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=WM6PZXNSY1HT9YW18SDP&pd_rd_wg=2JLxg&pd_rd_r=115d431c-0080-4579-873c-a1ca0800ff9a&pd_rd_i=B0DFQHSNF7", "https://www.amazon.in/Fargo-Leatherette-Shoulder-Womens-Yellow_FGO-638/dp/B0CBCPK1CX/ref=pd_ci_mcx_pspc_dp_d_2_hxwCMP_sspa_dk_detail_t_2?pd_rd_w=RAkmI&content-id=amzn1.sym.028be466-7201-4bae-831f-b191e6131def&pf_rd_p=028be466-7201-4bae-831f-b191e6131def&pf_rd_r=WM6PZXNSY1HT9YW18SDP&pd_rd_wg=2JLxg&pd_rd_r=115d431c-0080-4579-873c-a1ca0800ff9a&pd_rd_i=B0CBCPK1CX"]
    #urls = ['https://www.amazon.in/Fargo-Stylish-Textured-Sling-Ladies/product-reviews/B0DFQHSNF7/ref=cm_cr_getr_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=']
    #parent_xpath = "/html/body/div[2]/div/div[5]/div[27]/div/div/div/div/div[2]/div/div[2]/span[1]/div/div/div[3]/div[3]/div"
    #child_xpath_template = "/html/body/div[2]/div/div[5]/div[27]/div/div/div/div/div[2]/div/div[2]/span[1]/div/div/div[3]/div[3]/div/div[{index}]/div/div/div[4]/span/div/div[1]/span"

    # Call the function
    #final_result = []
    #N = 9
    #for url in urls:
    #  for c in range(1, N):
    #    url_str = url + str(c)
    #    print(f"Scraping {url_str}")
    #    results = extract_text_from_all_elements(url_str, parent_xpath, child_xpath_template)
    #    final_result.append(results)

    #    reviews = []
    #    print("Reviews are : ")
    #    for child_xpath, text in results['child_texts'].items():
    #        print(text)
    #        reviews.append(text)

    #    time.sleep(10)
  
  
    # Predefined taxonomy with keywords

    detailed_taxonomy = {
    "Design and Style": {
        "Trendy": {
            "Positive": ["stylish", "on-trend", "modern"],
            "Negative": ["outdated", "unfashionable"]
        },
        "Classic": {
            "Positive": ["timeless", "elegant"],
            "Negative": ["boring", "old-fashioned"]
        },
        "Color": {
            "Positive": ["vibrant", "true to picture"],
            "Negative": ["dull", "different from image"]
        },
        "Style": {
            "Positive": ["unique", "matches my taste"],
            "Negative": ["generic", "not as expected"]
        },
        "Aesthetic": {
            "Positive": ["beautiful", "eye-catching"],
            "Negative": ["ugly", "unappealing"]
        },
        "Size": {
            "Positive": ["appropiate"],
            "Negative": ["too small", "too big"]
        }
    },
    "Material and Craftsmanship": {
        "Durable": {
            "Positive": ["long-lasting", "sturdy"],
            "Negative": ["worn out quickly", "fragile"]
        },
        "Leather": {
            "Positive": ["premium quality", "genuine leather"],
            "Negative": ["fake leather", "peeling"]
        },
        "Fabric": {
            "Positive": ["soft", "comfortable to touch"],
            "Negative": ["cheap material", "tears easily"]
        },
        "Stitching": {
            "Positive": ["well-stitched", "strong seams"],
            "Negative": ["loose threads", "poorly stitched"]
        },
        "Quality": {
            "Positive": ["high-quality", "luxurious"],
            "Negative": ["low-quality", "cheap feel"]
        },
        "Material": {
            "Positive": ["excellent material", "feels premium"],
            "Negative": ["inferior material", "synthetic smell"]
        }
    },
    "Functionality": {
        "Storage": {
            "Positive": ["spacious", "good compartments"],
            "Negative": ["limited space", "poor organization"]
        },
        "Pockets": {
            "Positive": ["handy pockets", "well-designed"],
            "Negative": ["too few", "awkward placement"]
        },
        "Lightweight": {
            "Positive": ["easy to carry", "not bulky"],
            "Negative": ["feels heavy", "too bulky"]
        },
        "Comfortable": {
            "Positive": ["ergonomic", "easy to wear"],
            "Negative": ["hurts shoulders", "uncomfortable straps"]
        },
        "Versatile": {
            "Positive": ["fits every occasion", "multi-use"],
            "Negative": ["too specific", "not adaptable"]
        }
    },
    "Pricing": {
        "Affordability": {
            "Positive": ["great value", "worth every penny"],
            "Negative": ["overpriced", "not worth the price"]
        },
        "Value for Money": {
            "Positive": ["excellent deal", "reasonable price"],
            "Negative": ["bad investment", "poor value", "too expensive", "not justified"]
        }
    },
    "Brand and Packaging": {
        "Brand Reputation": {
            "Positive": ["trusted brand", "well-known"],
            "Negative": ["unknown brand", "not reputable"]
        },
        "Packaging": {
            "Positive": ["well-packaged", "secure"],
            "Negative": ["poorly packed", "damaged in transit"]
        },
        "Unboxing": {
            "Positive": ["luxury feel", "exciting experience"],
            "Negative": ["disappointing", "messy presentation"]
        },
        "Dust Bag": {
            "Positive": ["included dust bag", "added protection"],
            "Negative": ["missing dust bag", "cheap cover"]
        }
    },
    "Delivery and Condition": {
        "Delivery Time": {
            "Positive": ["arrived on time", "fast shipping"],
            "Negative": ["late delivery", "missed deadlines"]
        },
        "Damaged": {
            "Positive": ["in perfect condition", "no damage"],
            "Negative": ["scratched", "torn on arrival"]
        },
        "Packaging Quality": {
            "Positive": ["secure packaging", "no issues"],
            "Negative": ["damaged packaging", "unprofessional"]
        }
    },
    "Customer Support": {
        "Responsive": {
            "Positive": ["quick response", "helpful support"],
            "Negative": ["unresponsive", "ignored inquiries"]
        },
        "Return": {
            "Positive": ["easy return", "smooth process"],
            "Negative": ["complicated return", "no return policy"]
        },
        "Exchange": {
            "Positive": ["hassle-free exchange", "quick replacement"],
            "Negative": ["issues with exchange", "no exchange option"]
        },
        "Warranty": {
            "Positive": ["clear warranty", "long-term coverage"],
            "Negative": ["no warranty", "limited coverage"]
            }
        }
    }

    #Collect reviews from the last column of csv. 1st row is column name
    reviews_1 = get_reviews("docs/Sling & Cross-Body Bags_fargo_2024-11-25 (1).csv")
    reviews_2 = get_reviews("docs/Sling & Cross-Body Bags_fargo_2024-11-25.csv")
    reviews = reviews_1 + reviews_2

    #print(f"Reviews are {reviews}")
    #Number of reviews to analyze
    N = 50
    
    # Analyze reviews
    review_analysis = analyze_reviews(reviews[:N], detailed_taxonomy)

    # Display results
    display_results(review_analysis)

    # Plot overall category chart
    plot_category_pie_chart(review_analysis, "category_div")

    # Plot subcategory charts
    plot_subcategory_pie_charts(review_analysis, "subcategory_div")

    #Plot subcategory reason charts
    plot_reason_pie_charts(review_analysis, "subcategory_reason_div")

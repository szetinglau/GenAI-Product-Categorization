# GenAI Product Categorization

This repository provides a solution for automating product categorization using Generative AI models. It leverages Azure AI Search and OpenAI's GPT-4 to classify products based on their descriptions and other attributes.

## Features

- **Automated Product Categorization**: Classifies products by analyzing descriptions and attributes using AI models.
- **Integration with Azure AI Search**: Utilizes Azure AI Search to enhance the accuracy of product classification.
- **GPT-4 Integration**: Employs OpenAI's GPT-4 for natural language understanding and classification tasks.

## Prerequisites

- **Python 3.11+**: Ensure Python is installed on your system.
- **Azure Account**: Access to Azure AI Search services.
- **OpenAI API Key**: Access to OpenAI's GPT-4 model.

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/szetinglau/GenAI-Product-Categorization.git
   cd GenAI-Product-Categorization
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:

   Rename `.env-sample` to `.env` and update the following variables:

   ```env
   AZURE_SEARCH_ENDPOINT=your_azure_search_endpoint
   AZURE_SEARCH_KEY=your_azure_search_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Prepare Your Data**: Ensure your product data is in the correct format, similar to the provided sample files (e.g., `October Items.xlsx`).

2. **Run the Categorization Script**:

   ```bash
   python pfg_product_categorization.py
   ```

3. **Review Results**: The script will output the categorized products and their respective categories.

## Future Enhancements

- **Error Handling**: Address occasional content filter errors during classification.
- **Result Verification**: Implement logging of search results and prompts for verification purposes.
- **Web Interface**: Develop a user-friendly web UI for file uploads and result visualization.
- **Deployment**: Create a deployable solution for broader use cases.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

For more information, visit the [repository](https://github.com/szetinglau/GenAI-Product-Categorization). 

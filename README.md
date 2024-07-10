# DocuMind: Your Intelligent Document Assistant 🧠

DocuMind is an advanced document processing and question-answering system that leverages the power of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide intelligent responses based on your documents and web content.

## Features 🌟

- **Document Processing**: Upload and process various document formats (TXT, DOCX, PDF).
- **Web Crawling**: Crawl websites and add their content to the knowledge base.
- **Intelligent Chatbot**: Ask questions about your documents and get informed responses.
- **RAG System**: Utilizes a Retrieval-Augmented Generation system for accurate and context-aware answers.
- **LaTeX Rendering**: Supports rendering of LaTeX equations in chat responses.
- **Code Highlighting**: Displays code snippets with syntax highlighting.

## Screenshots 📸
<img width="1724" alt="image" src="https://github.com/YorkieDev/DocuMind/assets/42919623/28143490-9185-404d-ba9a-1e8b35ca07f2">
<img width="322" alt="image" src="https://github.com/YorkieDev/DocuMind/assets/42919623/b529567c-a86f-4489-be43-e6c915c7ad6d">
<img width="1370" alt="image" src="https://github.com/YorkieDev/DocuMind/assets/42919623/3b949441-a21c-4454-8c74-c956e76aa966">





## Requirements 📋

- Python 3.8+
- Streamlit
- OpenAI API (compatible with LM Studio)
- FAISS
- PyPDF2
- docx2txt
- tiktoken
- latex2mathml
- And other dependencies listed in `requirements.txt`

## Setup 🚀

1. Clone the repository:
   ```
   git clone https://github.com/yorkiedev/documind.git
   cd documind
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up LM Studio:
   - Ensure LM Studio is running and accessible at `http://localhost:1234/v1`
   - Configure your LM Studio with appropriate models and presets

4. Run the Streamlit app:
   ```
   streamlit run ragcrawler.py
   ```

## Usage 📚

1. **Document Upload**:
   - Use the sidebar to upload your documents (TXT, DOCX, PDF formats supported).
   - The system will process and index the documents automatically.

2. **Web Crawling**:
   - Enter a URL in the sidebar and specify the maximum number of pages to crawl.
   - Click "Crawl and Add to RAG" to add the web content to your knowledge base.

3. **Chatting with DocuMind**:
   - Type your questions in the chat input at the bottom of the page.
   - DocuMind will provide answers based on the content of your documents and crawled web pages.

4. **Viewing Responses**:
   - Responses may include formatted text, LaTeX equations, and code snippets.
   - LaTeX equations are rendered inline for easy reading.
   - Code snippets are displayed with syntax highlighting.

5. **Managing Documents**:
   - Use the "Clear All Documents" button in the sidebar to reset your knowledge base.

## Customization 🛠️

- Modify the `EMBEDDING_MODEL` and `CHAT_MODEL` constants in the script to use different language models.
- Adjust the chunk size and overlap in the `DocumentProcessor.split_text` method to fine-tune document processing.
- Customize the system prompt in the main chat loop to change the AI's behavior and personality.

## Contributing 🤝

Contributions to DocuMind are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 👏

- Thanks to the Streamlit team for their amazing framework.
- Kudos to the developers of FAISS, PyPDF2, and other libraries used in this project.
- Special thanks to the LM Studio community for their support and resources.

---

Happy document processing and chatting with DocuMind! If you have any questions or run into issues, please open an issue on GitHub. Enjoy! 🎉

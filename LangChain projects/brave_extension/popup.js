document.addEventListener('DOMContentLoaded', function() {
    const questionInput = document.getElementById('question');
    const askButton = document.getElementById('askButton');
    const answerDiv = document.getElementById('answer');
    const loadingDiv = document.getElementById('loading');

    askButton.addEventListener('click', async function() {
        const question = questionInput.value.trim();
        if (!question) {
            alert('Please enter a question');
            return;
        }

        // Show loading state
        loadingDiv.style.display = 'block';
        answerDiv.textContent = '';
        askButton.disabled = true;

        try {
            // Get the current tab's URL
            const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
            const url = tab.url;

            // Call the Python backend
            const response = await fetch('http://localhost:5000/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url,
                    question: question
                })
            });

            const data = await response.json();
            
            if (data.error) {
                answerDiv.textContent = `Error: ${data.error}`;
            } else {
                answerDiv.textContent = data.answer;
            }
        } catch (error) {
            answerDiv.textContent = `Error: ${error.message}`;
        } finally {
            // Hide loading state
            loadingDiv.style.display = 'none';
            askButton.disabled = false;
        }
    });
}); 
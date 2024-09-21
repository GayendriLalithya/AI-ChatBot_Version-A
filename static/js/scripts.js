function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() !== "") {
        const chatBox = document.getElementById('chat-box');
        const userMessage = document.createElement('div');
        userMessage.classList.add('user-message');
        userMessage.innerText = userInput;
        chatBox.appendChild(userMessage);
        document.getElementById('user-input').value = '';

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput })
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = document.createElement('div');
            botMessage.classList.add('bot-message');
            botMessage.innerText = data.response;
            chatBox.appendChild(botMessage);

            // Automatically read the bot's response aloud
            fetch('/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: data.response })
            });
        });
    }
}

function readAloud() {
    const chatBox = document.getElementById('chat-box');
    const lastMessage = chatBox.lastElementChild.innerText;

    fetch('/speak', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: lastMessage })
    });
}

function startListening() {
    fetch('/listen', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.text) {
            document.getElementById('user-input').value = data.text;
            sendMessage(); // Automatically send the recognized text
        } else {
            alert(data.error);
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const chatArea = document.getElementById("chat-area");
    const chatMessages = document.getElementById("chat-messages");
    const welcomeScreen = document.getElementById("welcome-screen");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    function hideWelcomeScreen() {
        if (welcomeScreen.style.display !== "none") {
            welcomeScreen.style.display = "none";
        }
    }

    function appendMessage(text, sender, isMarkdown = false) {
        hideWelcomeScreen();

        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");

        const contentDiv = document.createElement("div");
        contentDiv.classList.add("message-content");

        if (isMarkdown) {
            contentDiv.innerHTML = marked.parse(text);
        } else {
            contentDiv.textContent = text;
        }
        const timeDiv = document.createElement("div");
        timeDiv.classList.add("message-time");
        const now = new Date();
        timeDiv.textContent = now.toLocaleTimeString('ru-RU', {
            hour: '2-digit',
            minute: '2-digit'
        });

        msgDiv.appendChild(contentDiv);
        msgDiv.appendChild(timeDiv);
        chatMessages.appendChild(msgDiv);

        chatArea.scrollTop = chatArea.scrollHeight;
        return msgDiv;
    }

    async function sendMessage(textToProcess = null) {
        const text = textToProcess !== null ? textToProcess : userInput.value.trim();
        if (!text) return;

        appendMessage(text, "user");
        userInput.value = "";

        const loadingMsg = appendMessage("Поиск информации...", "bot");
        loadingMsg.querySelector('.message-content').classList.add('typing-indicator');

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();

            loadingMsg.remove();
            appendMessage(data.reply, "bot", true);

        } catch (error) {
            loadingMsg.remove();
            appendMessage("❌ Ошибка соединения с сервером.", "bot");
            console.error(error);
        }
    }

    sendBtn.addEventListener("click", () => sendMessage());
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    window.setPrompt = function (text) {
        sendMessage(text);
    };
});
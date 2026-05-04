document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    function appendMessage(text, sender, isMarkdown = false) {
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");

        const contentDiv = document.createElement("div");
        contentDiv.classList.add("message-content");

        if (isMarkdown) {
            contentDiv.innerHTML = marked.parse(text);
        } else {
            contentDiv.textContent = text;
        }

        msgDiv.appendChild(contentDiv);
        chatBox.appendChild(msgDiv);

        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, "user");
        userInput.value = "";

        const loadingMsg = appendMessage("Печатает...", "bot");
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
            appendMessage("Ошибка соединения с сервером.", "bot");
            console.error("Ошибка:", error);
        }
    }

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
});
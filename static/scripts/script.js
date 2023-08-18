function uploadFile() {
    var fileInput = document.getElementById('uploadFile');
    if (fileInput.files.length <= 0) {
        alert('Select a file before uploading.');
        return;
    }

    var file = fileInput.files[0];

    var fileExtension = file.name.split('.').pop().toLowerCase();
    if (fileExtension !== 'pdf') {
        alert('Invalid file format!!\nOnly .pdf files are allowed.');
        return;
    }

    var formData = new FormData();
    formData.append('file', file, file.name);

    $('#loadingModal').modal('show');
    // Start the timer when the modal is shown
    $('#loadingModal').on('shown.bs.modal', function () {
        var startTime = new Date();
        var timer = setInterval(function() {
            var elapsedSeconds = Math.floor((new Date() - startTime) / 1000);
            $('#timer').text(elapsedSeconds);
            }, 1000);
            // Stop the timer and hide the modal when the AJAX request is complete
            $.ajax({
                url: '/pdfupload',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    clearInterval(timer);
                    $('#loadingModal').modal('hide');
                    if (data.status != 201){
                        alert("Error uploading file");
                        return;
                    }
                    alert('File uploaded successfully.');
                    botChat("Hi, I am your virtual assistant. You can ask me any question about the PDF you just uploaded", true);
                    //document.getElementById("reset-chat").disabled=false;
                },
                error: function() {
                    $('#loadingModal').modal('hide');
                    alert('Error uploading file');
                }
            });
    });
};

function userChat(){
    const inputText = document.getElementById("chat-input").value.trim();
    if (inputText == ''){
        return;
    };
    console.log(inputText);
    var chatContainer = document.getElementById("chat-container");
    // Create a new chat bubble
    var newChatBubble = document.createElement("div");
    newChatBubble.className = "chat-bubble chat-user";
    // Create the chat bubble content
    var chatBubbleHeader = document.createElement("p");
    chatBubbleHeader.innerHTML = '<i class="fas fa-user"></i>&nbsp; User';
    chatBubbleHeader.style.fontWeight = "bold";
    var chatBubbleContent = document.createElement("p");
    chatBubbleContent.innerText = inputText;
    // Append the chat bubble content to the chat bubble
    newChatBubble.appendChild(chatBubbleHeader);
    newChatBubble.appendChild(chatBubbleContent);
    // Append the new chat bubble to the chat messages container
    chatContainer.appendChild(newChatBubble);
    $('#chat-input').val("");
    //
    var chatBubbleTypingContent = document.createElement("p");
    chatBubbleTypingContent.className = "typed";
    chatBubbleTypingContent.innerText = "Please wait.....";
    chatBubbleTypingContent.style.fontSize = "15px";

    chatContainer.appendChild(chatBubbleTypingContent);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    document.getElementById("chat-send").disabled=true;
    sendToBot(inputText);
    return;
};

function sendToBot(inputText){
    $.ajax({
        url: "/pdfchat",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({"input_text": inputText}),
        success: function(response) {
        // Handle the response from the Flask endpoint
        let botResponse = response["response"];
        // Update the chat UI with the bot's response
        botChat(botResponse);
        document.getElementById("chat-container").scrollTop = document.getElementById("chat-container").scrollHeight;
        },
        error: function(response) {
            alert(response["error"]);
            //alert('Some Error Occurred. Try again');
            document.getElementById("chat-send").disabled=false;
        }
    });
};

function botChat(botResponse,first=false){
    //remove the loading screen
    if (first == false){
        let typedElements = document.getElementsByClassName("typed");
        let lastTypedElement = Array.from(typedElements).pop();
        lastTypedElement.remove();
    };
    //
    var chatContainer = document.getElementById("chat-container");
    var newChatBubble = document.createElement("div");
    newChatBubble.className = "chat-bubble chat-bot";
    var chatBubbleHeader = document.createElement("p");
    chatBubbleHeader.innerHTML = '<i class="fas fa-robot"></i>&nbsp; PDFBot';
    chatBubbleHeader.style.fontWeight = "bold";
    var chatBubbleContent = document.createElement("p");
    chatBubbleContent.className = "response";
    chatBubbleContent.innerText = botResponse;
    // Append the chat bubble content to the chat bubble
    newChatBubble.appendChild(chatBubbleHeader);
    newChatBubble.appendChild(chatBubbleContent);
    // Append the new chat bubble to the chat messages container
    chatContainer.appendChild(newChatBubble);
    document.getElementById("chat-send").disabled=false;
    return;
};

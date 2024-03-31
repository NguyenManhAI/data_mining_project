import React, { useState } from 'react';
import './ChatPage.css';
import { API_URL_LSTM, API_URL_SVC } from './costants/urls.js';
import positiveIcon from './images/positive.png'
import negativeIcon from './images/negative.png'
const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [currentModel, setCurrentModel] = useState('SVC'); // Thêm state cho model hiện tại
  const [showModelList, setShowModelList] = useState(false); // Thêm state để kiểm soát việc hiển thị danh sách model
  const [currentUrl, setCurrentUrl] = useState(API_URL_SVC);
  const [sliderValue, setSliderValue] = useState(0); // Giá trị mặc định của thanh trượt
  const sendDatatoBackend = async (message) => {
    try {
      console.log(message, currentUrl);
      const response = await fetch(currentUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key: 'successfully!!!', message }),
      });
      const result = await response.json();
      // Hiển thị số nếu tồn tại
      const messageText = result.message.replace('negative', '<img src="' + negativeIcon + '" alt="Positive Icon" className="icon" />').replace('positive', '<img src="' + positiveIcon + '" alt="Positive Icon" className="icon" />');
      const proba = result.proba
      const label = result.label
      setMessages(prevMessages => [...prevMessages, { text: messageText,proba:proba, sender: 'bot',label : label}]);
    } catch (error) {
      console.log('Error:', error);
    }
  };

  const handleSendMessage = () => {
    if (newMessage.trim() !== '') {
      setMessages([...messages, { text: newMessage, sender: 'user' }]);
      sendDatatoBackend(newMessage)
      setNewMessage('');
    }
  };

  const handleModelSwitch = () => {
    // Thay đổi trạng thái của showModelList khi người dùng nhấn nút switch
    setShowModelList(!showModelList);
  };

  const handleModelSelect = (selectedModel) => {
    // Chọn model khi người dùng chọn từ dropdown menu
    setCurrentModel(selectedModel);
    // Ẩn danh sách model sau khi đã chọn
    setShowModelList(false);
    // thay đổi đường dẫn đến model đã chọn
    if(selectedModel === 'SVC' ){
      setCurrentUrl(API_URL_SVC)
    }
    else{
      setCurrentUrl(API_URL_LSTM)
    }
    console.log(selectedModel)
  };
  const handleSliderChange = (event) => {
    setSliderValue(event.target.value);
  };
  // function chỉ cần nhấn enter là gửi 
  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  };


  return (
    <div className="ChatApp">
      <div className="message-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
            <div className={`message-sender ${message.sender === 'user' ? 'user-sender' : 'bot-sender'}`}>
              {message.sender === 'user' ? 'You ' : 'AI-detect'}
            </div>
            <div className={`message-text ${message.sender === 'user' ? 'user-text' : 'bot-text'}`} dangerouslySetInnerHTML={{__html: message.text}}></div>
            {message.sender === 'bot' && ( // Hiển thị thanh trượt nếu tin nhắn từ bot
              <div >
                <input
                className= {`${message.label === 'positive' ? 'positive-slider' : 'negative-slider'}`}
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={message.proba }
                  onChange={handleSliderChange}
                /> {message.proba }%
                <span></span>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="input-container">
        <input
        className='input-at-container'
          type="text"
          placeholder="Type a message..."
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          onKeyDown={handleKeyDown} // Gọi hàm handleKeyDown khi người dùng nhấn enter
        />
        {/* Nút switch để hiển thị hoặc ẩn danh sách model */}
        <div >
          <button onClick={handleModelSwitch}>
            Model ({currentModel })
          </button>
          {/* Hiển thị danh sách model nếu showModelList là true */}
          {showModelList && (
            <div className="model-list">
              <button onClick={() => handleModelSelect('SVC')}>SVC Model</button>
              <button onClick={() => handleModelSelect('LSTM')}>LSTM Model</button>
              {/* Thêm các button khác tương ứng với các model khác nếu cần */}
            </div>
          )}
        </div>
        <button onClick={handleSendMessage}><span role="img" aria-label="Send">
          ✉️
        </span></button>

      </div>
    </div>
  );
};

export default ChatPage;

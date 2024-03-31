import React, { useState } from 'react';
import './RegistrationForm.css'
function RegistrationForm() {

    const [userName, setuserName] = useState(null);
    const [email, setEmail] = useState(null);
    const [password, setPassword] = useState(null);
    const [confirmPassword, setConfirmPassword] = useState(null);

    const sendDatatoBackend = async () => {
        try {
          
          const response = await fetch('http://localhost:5000/register', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ key: 'successfully!!!',userName, email, password, confirmPassword}),
          });
          const result = await response.json();   // response from server
          
        } catch (error) {
          console.log('Error:', error);
        }
      };

    const handleInputChange = (e) => {
        const { id, value } = e.target;
        if (id === "userName") {
            setuserName(value);
        }
        if (id === "email") {
            setEmail(value);
        }
        if (id === "password") {
            setPassword(value);
        }
        if (id === "confirmPassword") {
            setConfirmPassword(value);
        }

    }

    const handleSubmit = () => {
        console.log(userName, email, password, confirmPassword);
        sendDatatoBackend();
    }

    return (
        <div className="form">
            <div className="form-body">
                <div className="username">
                    <label className="form__label" for="userName">User Name </label>
                    <input className="form__input" type="text" value= {userName} onChange = {(e) => handleInputChange(e)} id="userName" placeholder="User Name" />
                </div>
                <div className="email">
                    <label className="form__label" for="email">Email </label>
                    <input type="email" id="email" className="form__input" value= {email} onChange = {(e) => handleInputChange(e)} placeholder="Email" />
                </div>
                <div className="password">
                    <label className="form__label" for="password">Password </label>
                    <input className="form__input" type="password" id="password" value= {password} onChange = {(e) => handleInputChange(e)} placeholder="Password" />
                </div>
                <div className="confirm-password">
                    <label className="form__label" for="confirmPassword">Confirm Password </label>
                    <input className="form__input" type="password" id="confirmPassword" value= {confirmPassword} onChange = {(e) => handleInputChange(e)} placeholder="Confirm Password" />
                </div>
            </div>
            <div class= "footer">
                <button onClick={()=>handleSubmit()} type="submit" class="btn">Register</button>
            </div>
        </div>
    )
}
export default RegistrationForm;
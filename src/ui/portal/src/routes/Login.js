import {useState} from "react";

const Login = () => {
    const [userName, setUserName] = useState("");
    const [password, setPassword] = useState("");

    const handleButtonClick = async (e) => {
        const rawResponse = await fetch('http://192.168.0.102:5000/api/token', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({userName, email: "", password})
        })
            .then(response => response.json())
            .then(data => console.log(data.result));
    }

    return (
      <>
          <h2>Login</h2>
          <input placeholder="Username" type="text" value={userName} onChange={text => setUserName(text.target.value)}/>
          <br/>
          <input placeholder="Password" type="password" value={password} onChange={text => setPassword(text.target.value)}/>
          <br/>
          <button onClick={e => handleButtonClick(e)}>Log in</button>
      </>
    );
}

export default Login;
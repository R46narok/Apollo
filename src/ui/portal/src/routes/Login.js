import {useState} from "react";
import {useDispatch, useSelector} from "react-redux";
import {userActions} from "../actions/userActions";

const Login = () => {
    const [userName, setUserName] = useState("");
    const [password, setPassword] = useState("");
    const [text, setText] = useState("");

    const dispatch = useDispatch();
    const handleButtonClick = async (e) => {
        dispatch(userActions.login(userName, password))
    }

    return (
      <>
          <h2>Login</h2>
          <input placeholder="Username" type="text" value={userName} onChange={text => setUserName(text.target.value)}/>
          <br/>
          <input placeholder="Password" type="text" value={password} onChange={text => setPassword(text.target.value)}/>
          <br/>
          <button onClick={e => handleButtonClick(e)}>Log in</button>
          <br/>

      </>
    );
}

export default Login;
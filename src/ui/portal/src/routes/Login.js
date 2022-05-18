import {useState} from "react";
import {useDispatch, useSelector} from "react-redux";
import {userActions} from "../actions/userActions";
import "../bootstrap.min.css"
import LoginImage from './LoginImage.svg';
import { useNavigate } from 'react-router-dom';

function Login() {
    const [userName, setUserName] = useState("");
    const [password, setPassword] = useState("");
    const [text, setText] = useState("");

    const navigate = useNavigate();
    const dispatch = useDispatch();
    const handleButtonClick = async (e) => {
        e.preventDefault()
        dispatch(userActions.login(userName, password));
        navigate("/");
    }

    return (
        <div className="content" style={{paddingTop: 50}}>
            <div className="container">
                <div className="row">
                    <div className="col-md-6">
                        <img src={LoginImage} className="img-fluid" alt="Image"/>
                    </div>
                    <div className="col-md-6 contents">
                        <div className="row justify-content-center">
                            <div className="col-md-8">
                                <div className="mb-4">
                                    <h3>Sign In</h3>
                                    <p className="mb-4">Sign into your account to harden your system's security.</p>
                                </div>
                                <form onSubmit={handleButtonClick}>
                                    <div className="form-group first">
                                        <label htmlFor="username">Username</label>
                                        <input type="text" className="form-control" id="username" onChange={(e) => setUserName(e.target.value)}/>

                                    </div>
                                    <div className="form-group last mb-4">
                                        <label htmlFor="password">Password</label>
                                        <input type="password" className="form-control" id="password" onChange={(e) => setPassword(e.target.value)}/>

                                    </div>

                                    <div className="d-flex mb-5 align-items-center">
                                        <span className="ml-auto"><a href="#" className="forgot-pass">Forgot Password</a></span>
                                    </div>

                                    <input type="submit" value="Log In" className="btn btn-block btn-primary"/>
                                        <span className="d-block text-left my-4 text-muted">&mdash; or login with &mdash;</span>

                                        <div className="social-login">
                                            <a href="#" className="facebook">
                                                <span className="icon-facebook mr-3"></span>
                                            </a>
                                            <a href="#" className="twitter">
                                                <span className="icon-twitter mr-3"></span>
                                            </a>
                                            <a href="#" className="google">
                                                <span className="icon-google mr-3"></span>
                                            </a>
                                        </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Login;
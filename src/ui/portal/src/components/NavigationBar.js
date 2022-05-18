import "../bootstrap.min.css"
import {useSelector} from "react-redux";
import {NavLink, useHistory} from "react-router-dom";

const NavigationBar = ({active}) => {
    const baseStyle = "nav-link";
    const activeStyle = "nav-link active";

    const homeStyle = active === "Home" ? activeStyle : baseStyle;

    const authStatus = useSelector((state) => state.authentication);

    return(
        <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
            <div className="container-fluid">
                <a className="navbar-brand" href="#">Apollo</a>
                <button className="navbar-toggler" type="button" data-bs-toggle="collapse"
                        data-bs-target="#navbarColor01" aria-controls="navbarColor01" aria-expanded="false"
                        aria-label="Toggle navigation">
                    <span className="navbar-toggler-icon"></span>
                </button>

                <div className="collapse navbar-collapse" id="navbarColor01">
                    <ul className="navbar-nav me-auto">
                        <li className="nav-item">
                            <a className={homeStyle} href="#">Home
                                <span className="visually-hidden">(current)</span>
                            </a>
                        </li>

                        <li className="nav-item">
                            <a className="nav-link" href="#">Features</a>
                        </li>
                        {
                            (authStatus.loggedIn) ?
                                <>
                                    <li className="nav-item">
                                        <a className="nav-link" href="#">Profile</a>
                                    </li>
                                    <li className="nav-item dropdown">
                                        <a className="nav-link dropdown-toggle" data-bs-toggle="dropdown" href="#" role="button"
                                           aria-haspopup="true" aria-expanded="false">Portal</a>
                                        <div className="dropdown-menu">
                                            <a className="dropdown-item" href="#">Action</a>
                                            <a className="dropdown-item" href="#">Another action</a>
                                            <a className="dropdown-item" href="#">Something else here</a>
                                            <div className="dropdown-divider"></div>
                                            <a className="dropdown-item" href="#">Separated link</a>
                                        </div>
                                    </li>
                                </>
                                :
                                <>

                                    <li className="nav-item">
                                        <NavLink
                                            to="/profile/login"
                                            className="nav-link"
                                        >
                                            Login
                                        </NavLink>
                                    </li>
                                </>
                        }
                    </ul>
                    <form className="d-flex">
                        <input className="form-control me-sm-2" type="text" placeholder="Search"/>
                        <button className="btn btn-secondary my-2 my-sm-0" type="submit">Search</button>
                    </form>
                </div>
            </div>
        </nav>
    );
}

export default NavigationBar;
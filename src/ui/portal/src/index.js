import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import {
    BrowserRouter,
    Routes,
    Route,
} from "react-router-dom";
import {Provider} from "react-redux";
import Login from "./routes/Login";
import {store} from "./helpers/store";
import NavigationBar from "./components/NavigationBar";
import Footer from "./components/Footer";


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
      <Provider store={store}>
          <BrowserRouter>
              <NavigationBar active="Home"/>
              <Routes>
                  <Route path="/" element={<App />} />
                  <Route path="profile/login" element={<Login/>} />
              </Routes>
              <Footer/>
          </BrowserRouter>
      </Provider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

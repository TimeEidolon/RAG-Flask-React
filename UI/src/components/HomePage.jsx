import React from 'react';
import '../assets/css/HomePage.css'
import viteLogo from '../assets/img/vite.svg';
import reactLogo from '../assets/img/react.svg';

const HomePage = () => (
    <div className="App">
        <div>
            <a href="https://vitejs.dev" target="_blank">
                <img src={viteLogo} className="logo" alt="Vite logo"/>
            </a>
            <a href="https://reactjs.org" target="_blank">
                <img src={reactLogo} className="logo react" alt="React logo"/>
            </a>
        </div>
        <h1>Rag-Flask-React</h1>
    </div>
);

export default HomePage;
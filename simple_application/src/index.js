import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter as Router,
Route, Link, Switch} from 'react-router-dom';
import SignList from "./signList";
import Detection from "./detection";
import "./styles.css";

class App extends React.Component {
  render() {
    return (
    <Router>
      <div className="main-container">
        <header className="header">
            <p className="header-text">
               <Link to="/"> Detection page </Link>
            </p>
            <p className="header-text">
                <Link to="/sign-list"> List of signs </Link>
            </p>
        </header>
        <Switch>
            <Route path="/sign-list">
                <SignList/>
            </Route>
            <Route path="/">
                <Detection/>
            </Route>
        </Switch>
        </div>
      </Router>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
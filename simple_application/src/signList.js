import React from "react";
import signsObjects from "./signsObjects";

class SignList extends React.Component {
    render(){
        return (
            <div className="signs-container">
                { Object.keys(signsObjects).map(sign => (
                    <div className="sign-info">
                        <img src={ signsObjects[sign].imgLink } alt={ signsObjects[sign].name }></img>
                        <p className="sign-name">{ signsObjects[sign].name }</p>
                    </div>
                )) }
            </div>
        )
    }
}

export default SignList;
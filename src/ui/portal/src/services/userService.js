export const userService = {
    login,
    logout
}

function parseJwt(token) {
    var base64Url = token.split('.')[1];
    var base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    var jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

    return JSON.parse(jsonPayload);
}

function login(userName, password) {
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({userName, password, email: ""})
    }

    return fetch("http://localhost:5000/api/token", requestOptions)
        .then(handleResponse)
        .then(token => {
            localStorage.setItem("jwt", token);
            var parsed = parseJwt(token);
            var user = {
                userName: parsed["http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name"],
                role: parsed["http://schemas.microsoft.com/ws/2008/06/identity/claims/role"]
            };
            localStorage.setItem("user", user);

            return user;
        });
}

function logout() {

}

function handleResponse(response) {
    return response.text().then(text => {
        const data = text && JSON.parse(text);
        if (!response.ok) {
            if (response.status === 401) {
                logout();
            }

            const error = (data && data.message) || response.statusText;
            return Promise.reject(error);
        }

        return data.result;
    });
}
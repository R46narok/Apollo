import {userConstants} from "../constants/userConstants";
import {userService} from "../services/userService";

export const userActions = {
    login,
    logout
}

function login(userName, password) {
    return dispatch => {
        dispatch(request({userName}));
        userService.login(userName, password)
            .then(user => {
                    dispatch(success(user));
                },
                error => {
                    dispatch(failure(error));
                });
    }

    function request(user) { return { type: userConstants.LOGIN_REQUEST, user } }
    function success(user) { return { type: userConstants.LOGIN_SUCCESS, user } }
    function failure(error) { return { type: userConstants.LOGIN_FAILURE, error } }
}

function logout() {
    userService.logout();
    return { type: userConstants.LOGOUT };
}
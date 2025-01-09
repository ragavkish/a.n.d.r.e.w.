document.addEventListener('DOMContentLoaded', () => {
    const emailContainer = document.getElementById('email-container');
    const passwordContainer = document.getElementById('password-container');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const showPasswordButton = document.getElementById('show-password-button');

    function createDynamicInput(container, hiddenInput, isPassword = false) {
        const boxes = [];
    
        function createBox() {
            const box = document.createElement('input');
            box.type = 'text';
            box.maxLength = 1;
            box.classList.add('input-box');
            
            if (isPassword) {
                box.style.webkitTextSecurity = 'disc';
            } else {
                box.style.webkitTextSecurity = 'none';
            }
    
            box.addEventListener('input', () => {
                if (box.value) {
                    updateHiddenInput();
                    createBox();
                }
            });
    
            box.addEventListener('keydown', (event) => {
                if (event.key === 'Backspace' && !box.value && boxes.length > 1) {
                    container.removeChild(box);
                    boxes.pop();
                    boxes[boxes.length - 1].focus();
                    updateHiddenInput();
                }
            });
    
            container.appendChild(box);
            boxes.push(box);
            box.focus();
        }
    
        function updateHiddenInput() {
            hiddenInput.value = boxes.map((b) => b.value).join('');
        }
    
        createBox();
    }    

    createDynamicInput(emailContainer, emailInput, false);
    createDynamicInput(passwordContainer, passwordInput, true);

    showPasswordButton.addEventListener('click', () => {
        const passwordBoxes = document.querySelectorAll('#password-container .input-box');
        const isHidden = passwordBoxes[0].style.webkitTextSecurity === 'disc';

        passwordBoxes.forEach(box => {
            box.style.webkitTextSecurity = isHidden ? 'none' : 'disc';
        });

        showPasswordButton.textContent = isHidden ? '(hide pa$$)' : '(show pa$$)';
    });
});

document.getElementById('enter-button').addEventListener('click', function() {
    document.querySelector('form').submit();
});

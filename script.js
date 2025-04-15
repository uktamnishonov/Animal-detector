function showLiveCamera() {
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
      })
      .catch(err => {
        alert("Could not access camera: " + err);
      });
  
    // Dummy data simulation
    setInterval(() => {
      document.getElementById('animalName').innerText = "Deer";
      document.getElementById('accuracy').innerText = Math.floor(Math.random() * 100) + "%";
    }, 2000);
  }
  
  function uploadPhoto() {
    const fileInput = document.getElementById('fileInput');
    fileInput.accept = "image/*";
    fileInput.click();
    fileInput.onchange = () => {
      alert("Image uploaded! (Backend will process this)");
    };
  }
  
  function uploadVideo() {
    const fileInput = document.getElementById('fileInput');
    fileInput.accept = "video/*";
    fileInput.click();
    fileInput.onchange = () => {
      alert("Video uploaded! (Backend will process this)");
    };
  }

  // JavaScript to toggle dark mode
  const modeToggleBtn = document.getElementById('modeToggle');

  modeToggleBtn.addEventListener('click', function () {
    document.body.classList.toggle('dark-mode');
  
    if (document.body.classList.contains('dark-mode')) {
      this.textContent = '🌞 Light Mode';
    } else {
      this.textContent = '🌑 Dark Mode';
    }
  }
);
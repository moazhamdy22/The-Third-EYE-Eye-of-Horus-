# audio_feedback_vision_assitant.py
import pyttsx3
import threading
import queue
import logging
import time

# Configure a logger for this module
logger = logging.getLogger(__name__) # Will inherit config from main.py

# Define command types for clarity
COMMAND_SPEAK = "SPEAK"
COMMAND_FORCE_STOP = "FORCE_STOP"
COMMAND_SHUTDOWN = "SHUTDOWN"
# No new command needed for speak_blocking, it uses COMMAND_SPEAK + polling

class AudioFeedbackHandler:
    def __init__(self, rate=160, volume=1.0):
        logger.info("Initializing AudioFeedbackHandler...")
        self.engine = None
        self.message_queue = queue.Queue()
        self.speaking_flag_lock = threading.Lock() # Lock for self.speaking
        self._speaking = False # Internal flag, use property for access
        self._stop_event = threading.Event()
        self.speak_thread = None

        try:
            self.engine = pyttsx3.init()
            if self.engine: # Check if init was successful
                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume)
                self.engine.say(" ") # Test speech
                self.engine.runAndWait()
                logger.info(f"pyttsx3 engine initialized and tested. Driver: {getattr(self.engine.proxy, '__class__', 'Unknown').__name__}")
            else:
                logger.critical("pyttsx3.init() failed to return an engine instance.")
                return
        except Exception as e:
            logger.critical(f"Failed to initialize pyttsx3 engine: {e}", exc_info=True)
            self.engine = None
            return

        self._start_speak_thread()
        logger.info("AudioFeedbackHandler initialized and worker thread started.")

    @property
    def speaking(self):
        with self.speaking_flag_lock:
            return self._speaking

    @speaking.setter
    def speaking(self, value):
        with self.speaking_flag_lock:
            self._speaking = value

    def _start_speak_thread(self):
        if self.speak_thread is None or not self.speak_thread.is_alive():
            self._stop_event.clear()
            self.speak_thread = threading.Thread(target=self._process_queue, name="AudioWorkerThread", daemon=True)
            self.speak_thread.start()
            logger.debug("Audio processing thread (re)started.")
        else:
            logger.debug("Audio processing thread already running.")

    def _process_queue(self):
        logger.debug(f"{threading.current_thread().name} loop started.")
        while not self._stop_event.is_set():
            try:
                command_type, message_content = self.message_queue.get(timeout=0.1)

                if command_type == COMMAND_SHUTDOWN:
                    logger.debug(f"{threading.current_thread().name} received SHUTDOWN.")
                    # If the worker was in the middle of speaking when shutdown was queued
                    if self.engine and self.speaking: # Use property for thread-safe check
                        logger.info(f"{threading.current_thread().name} was speaking during SHUTDOWN, attempting engine.stop().")
                        try:
                            self.engine.stop()
                        except RuntimeError as e: # pyttsx3 can raise this if engine is in a bad state
                            logger.error(f"{threading.current_thread().name} RuntimeError during SHUTDOWN engine.stop(): {e}", exc_info=True)
                        except Exception as e:
                             logger.error(f"{threading.current_thread().name}  Unexpected error during SHUTDOWN engine.stop(): {e}", exc_info=True)
                    self.message_queue.task_done()
                    logger.debug(f"{threading.current_thread().name} exiting loop due to SHUTDOWN.")
                    break # Exit the loop

                if not self.engine:
                    logger.error(f"{threading.current_thread().name}: TTS engine not available. Discarding command.")
                    self.message_queue.task_done()
                    continue

                if command_type == COMMAND_SPEAK:
                    if self._stop_event.is_set(): # Check before long operation
                        logger.info(f"{threading.current_thread().name} received SPEAK but stop_event is set. Discarding.")
                        self.message_queue.task_done()
                        continue

                    logger.info(f"{threading.current_thread().name} speaking: '{message_content[:60]}...'")
                    self.speaking = True
                    try:
                        self.engine.say(message_content)
                        self.engine.runAndWait() # This is blocking
                    except RuntimeError as e:
                        logger.error(f"{threading.current_thread().name} RuntimeError during say/runAndWait: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"{threading.current_thread().name}  Unexpected error during speech: {e}", exc_info=True)
                    finally:
                        self.speaking = False # Critical to reset this
                    logger.debug(f"{threading.current_thread().name} SPEAK command finished for '{message_content[:30]}...'.")

                elif command_type == COMMAND_FORCE_STOP:
                    logger.info(f"{threading.current_thread().name} processing FORCE_STOP command.")
                    if self.speaking and self.engine: # Use property, check engine
                        logger.debug(f"{threading.current_thread().name} calling engine.stop() due to FORCE_STOP command.")
                        try:
                            self.engine.stop()
                        except RuntimeError as e:
                            logger.error(f"{threading.current_thread().name} RuntimeError during worker engine.stop(): {e}", exc_info=True)
                        except Exception as e:
                             logger.error(f"{threading.current_thread().name}  Unexpected error during worker engine.stop(): {e}", exc_info=True)
                        # self.speaking should become False when runAndWait() exits in the SPEAK block
                    else:
                        logger.debug(f"{threading.current_thread().name} not actively speaking or no engine; engine.stop() in worker skipped for FORCE_STOP.")

                    logger.debug(f"{threading.current_thread().name} clearing pending SPEAK messages after FORCE_STOP.")
                    temp_keep_queue = []
                    while not self.message_queue.empty():
                        try:
                            p_cmd, p_msg = self.message_queue.get_nowait()
                            if p_cmd == COMMAND_SPEAK:
                                logger.debug(f"Discarded pending SPEAK by worker after FORCE_STOP: '{p_msg[:30]}...'")
                            else:
                                temp_keep_queue.append((p_cmd, p_msg))
                            self.message_queue.task_done() # Mark task done for get_nowait too
                        except queue.Empty:
                            break
                    for item in temp_keep_queue:
                        self.message_queue.put(item)
                    logger.debug(f"{threading.current_thread().name} FORCE_STOP processing finished.")

                self.message_queue.task_done()

            except queue.Empty:
                continue 
            except Exception as e:
                logger.error(f"Critical error in {threading.current_thread().name} loop: {e}", exc_info=True)
                # Potentially set speaking to False here if a critical error occurs mid-speech?
                # self.speaking = False # Risky if not sure about state
                time.sleep(0.1) 

        logger.info(f"{threading.current_thread().name} loop finished.")
        self.speaking = False # Ensure flag is reset on exit

    def speak(self, message: str):
        if not isinstance(message, str) or not message.strip():
            logger.warning(f"Speak called with invalid message: '{message}' by {threading.current_thread().name}")
            return

        if not self.engine:
            logger.error(f"Cannot speak, TTS engine not available (called by {threading.current_thread().name}).")
            return

        if self._stop_event.is_set() or not (self.speak_thread and self.speak_thread.is_alive()):
            logger.warning(f"Audio handler shutting/shut down or thread dead, not queueing: '{message[:30]}...'")
            return

        logger.debug(f"Queueing SPEAK: '{message[:60]}...' by {threading.current_thread().name}")
        self.message_queue.put((COMMAND_SPEAK, message))

    def speak_blocking(self, message: str, timeout_per_word=0.35, base_timeout=1.5, pre_stop_delay=0.3):
        """
        Speaks a message and blocks the CALLING thread until speech is likely finished or a timeout occurs.
        Uses the standard SPEAK command and polls the 'speaking' flag.
        This version first waits for any current speech to finish, then interrupts if necessary.
        """
        if not isinstance(message, str) or not message.strip():
            logger.warning(f"Speak_blocking called with invalid message: '{message}' by {threading.current_thread().name}")
            return

        if not self.engine:
            logger.error(f"Cannot speak_blocking, TTS engine not available (called by {threading.current_thread().name}).")
            return

        if self._stop_event.is_set() or not (self.speak_thread and self.speak_thread.is_alive()):
            logger.warning(f"Audio handler shutting/shut down or thread dead, not queueing for speak_blocking: '{message[:30]}...'")
            return

        # Wait for any *current* speech to finish before queueing this blocking message,
        # but only for a short period. If still speaking after that, interrupt.
        wait_for_clear_start = time.time()
        # Wait up to, e.g., 2 seconds for current speech to end naturally.
        # Adjust this timeout as needed. A very short timeout makes it more interruptive.
        max_wait_for_current_speech = 2.0 
        while self.speaking and (time.time() - wait_for_clear_start < max_wait_for_current_speech):
            if self._stop_event.is_set(): 
                logger.info("Speak_blocking: Stop event detected while waiting for current speech to clear.")
                return
            time.sleep(0.05)
        
        if self.speaking: # If still speaking after the grace period, force stop it.
            logger.info(f"Speak_blocking: Current speech ongoing after {max_wait_for_current_speech}s. Forcing stop to prioritize '{message[:30]}...'.")
            self.force_stop()
            time.sleep(pre_stop_delay) # Give force_stop time to take effect and worker to clear queue

        logger.debug(f"Queueing SPEAK (for speak_blocking): '{message[:60]}...' by {threading.current_thread().name}")
        self.message_queue.put((COMMAND_SPEAK, message)) # Now queue the blocking message

        num_words = len(message.split())
        estimated_timeout = base_timeout + (num_words * timeout_per_word)
        start_wait_time = time.time()
        
        speech_started_for_this_message = False
        # Wait for *this* message to begin (or for self.speaking to become true)
        # This timeout should be relatively short, just to ensure the worker picked it up.
        wait_for_start_timeout = base_timeout # Can use a separate shorter timeout if needed
        while time.time() - start_wait_time < wait_for_start_timeout: 
            if self._stop_event.is_set(): 
                logger.info("Speak_blocking: Stop event detected while waiting for this message to start.")
                return
            if self.speaking: # Our message (or any other) has started.
                speech_started_for_this_message = True
                break
            time.sleep(0.05)

        if speech_started_for_this_message:
            logger.debug(f"Speak_blocking: Message '{message[:30]}...' appears to have started. Waiting for completion (overall timeout: {estimated_timeout:.2f}s).")
            # Now wait for self.speaking to become False, indicating speech completion.
            while self.speaking and (time.time() - start_wait_time < estimated_timeout):
                if self._stop_event.is_set():
                     logger.info("Speak_blocking: Stop event detected while waiting for this message to complete.")
                     if self.speaking: self.force_stop() # Stop if it was cut short by shutdown
                     return
                time.sleep(0.05)
            
            if self.speaking: # Still speaking after the full estimated_timeout for this message
                logger.warning(f"Speak_blocking: Timed out waiting for '{message[:30]}...' to finish after it started. Forcing stop.")
                self.force_stop()
            else:
                logger.debug(f"Speak_blocking: Message '{message[:30]}...' assumed finished (speaking flag is False).")
        else:
            logger.warning(f"Speak_blocking: Message '{message[:30]}...' did not appear to start speaking within {wait_for_start_timeout}s.")
            # If it didn't start, it might be stuck in the queue behind a very long message that wasn't interrupted,
            # or the worker thread has an issue. A force_stop here might clear the queue.
            self.force_stop() # Attempt to clear any blockage if it seems stuck before starting.


    def force_stop(self):
        """
        Attempts an immediate stop of current speech and clears pending speech.
        Safe to call from any thread.
        """
        logger.info(f"Force_stop called by {threading.current_thread().name}")
        if not self.engine:
            logger.warning("Cannot force_stop, TTS engine not available.")
            return

        if self.speaking: 
            logger.debug(f"Attempting immediate engine.stop() from {threading.current_thread().name} due to force_stop.")
            try:
                self.engine.stop()
            except RuntimeError as e:
                 logger.warning(f"RuntimeError during immediate engine.stop() in force_stop from {threading.current_thread().name}: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Error during immediate engine.stop() in force_stop from {threading.current_thread().name}: {e}", exc_info=True)
        else:
            logger.debug(f"Not actively speaking, immediate engine.stop() in force_stop skipped by {threading.current_thread().name}.")

        logger.debug(f"Queueing COMMAND_FORCE_STOP for worker by {threading.current_thread().name}.")
        self.message_queue.put((COMMAND_FORCE_STOP, ""))

    def stop(self):
        """
        Signals the audio processing thread to shut down gracefully.
        This should be called when the application is closing.
        Revised to be gentler and rely more on worker thread for shutdown.
        """
        caller_thread_name = threading.current_thread().name
        logger.info(f"Stop called for AudioFeedbackHandler by {caller_thread_name}.")

        if self._stop_event.is_set():
            logger.info(f"AudioFeedbackHandler already signaled to stop (called by {caller_thread_name}).")
            if self.speak_thread and not self.speak_thread.is_alive():
                 logger.info(f"Worker thread already stopped (stop called by {caller_thread_name}).")
            return

        self._stop_event.set() # Signal worker thread to stop

        # Queue SHUTDOWN command. The worker should process this after its current item.
        # The worker's SHUTDOWN logic will attempt engine.stop() if it was speaking.
        logger.debug(f"Queueing SHUTDOWN command for worker by {caller_thread_name}.")
        try:
            self.message_queue.put((COMMAND_SHUTDOWN, ""), block=False) 
        except queue.Full:
            logger.warning(f"Could not queue SHUTDOWN (queue full for {self.speak_thread.name}). Trying blocking put.")
            try:
                self.message_queue.put((COMMAND_SHUTDOWN, ""), block=True, timeout=0.5)
            except queue.Full:
                logger.error(f"Still could not queue SHUTDOWN for {self.speak_thread.name}. Worker is likely unresponsive.")
                if self.engine:
                    logger.warning(f"Forcefully stopping engine from stop() due to inability to queue SHUTDOWN for {self.speak_thread.name}.")
                    try: self.engine.stop()
                    except Exception as e_stop_critical: 
                        logger.error(f"Critical engine.stop() error in stop() for {self.speak_thread.name}: {e_stop_critical}", exc_info=True)


        if self.speak_thread and self.speak_thread.is_alive():
            # This join waits for the worker thread to finish its current task 
            # (hopefully speaking a final message) and then process the SHUTDOWN command.
            short_join_timeout = 2.0 # Initial wait for graceful exit
            logger.debug(f"Waiting for {self.speak_thread.name} to join (initial timeout {short_join_timeout}s), called by {caller_thread_name}...")
            self.speak_thread.join(timeout=short_join_timeout)

            if self.speak_thread.is_alive():
                # If still alive, it might be stuck in runAndWait(), or SHUTDOWN command's stop wasn't enough.
                logger.warning(f"{self.speak_thread.name} did not join after initial timeout. Attempting engine.stop() from {caller_thread_name} as a fallback.")
                if self.engine:
                    try:
                        self.engine.stop() # Forceful stop from the calling thread
                    except RuntimeError as e:
                        logger.error(f"RuntimeError during engine.stop() in stop() for {self.speak_thread.name}: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Unexpected error during engine.stop() in stop() for {self.speak_thread.name}: {e}", exc_info=True)
                
                remaining_join_timeout = 3.0 # Give it a bit more time after the forceful stop
                logger.debug(f"Waiting for {self.speak_thread.name} to join again (timeout {remaining_join_timeout}s)...")
                self.speak_thread.join(timeout=remaining_join_timeout)

                if self.speak_thread.is_alive():
                    logger.error(f"{self.speak_thread.name} FAILED to join after all attempts. Called by {caller_thread_name}.")
                else:
                    logger.info(f"{self.speak_thread.name} joined after forceful stop attempts from {caller_thread_name}.")
            else:
                logger.info(f"{self.speak_thread.name} joined successfully within initial timeout. Called by {caller_thread_name}.")
        else:
            logger.info(f"Speak thread was not active or already joined when stop() was called by {caller_thread_name}.")
        
        if self.engine:
            logger.debug(f"Setting self.engine to None in stop() by {caller_thread_name}.")
            self.engine = None
        logger.info(f"AudioFeedbackHandler stop() finished by {caller_thread_name}.")

    def __del__(self):
        logger.debug(f"AudioFeedbackHandler __del__ called for instance {id(self)}.")
        if not self._stop_event.is_set(): # Check if stop was already called
            logger.warning(f"AudioFeedbackHandler instance {id(self)} was garbage collected without explicit stop(). Calling stop().")
            self.stop()
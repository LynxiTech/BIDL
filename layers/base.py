# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


class StepModule:
    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`

        .. _StepModule.supported_step_mode-cn:

        :return: 包含支持的后端的tuple
        :rtype: tuple[str]

        返回此模块支持的步进模式。

        * :ref:`中文 API <StepModule.supported_step_mode-cn>`

        .. _StepModule.supported_step_mode-en:

        :return: a tuple that contains the supported backends
        :rtype: tuple[str]

        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`

        .. _StepModule.step_mode-cn:

        :return: 模块当前使用的步进模式
        :rtype: str

        * :ref:`中文 API <StepModule.step_mode-cn>`

        .. _StepModule.step_mode-en:

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`

        .. _StepModule.step_mode-setter-cn:

        :param value: 步进模式
        :type value: str

        将本模块的步进模式设置为 ``value``

        * :ref:`中文 API <StepModule.step_mode-setter-cn>`

        .. _StepModule.step_mode-setter-en:

        :param value: the step mode
        :type value: str

        Set the step mode of this module to be ``value``

        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value

class SingleModule(StepModule):
    """
    * :ref:`API in English <SingleModule-en>`

    .. _SingleModule-cn:

    只支持单步的模块 (``step_mode == 's'``)。

    * :ref:`中文 API <SingleModule-cn>`

    .. _SingleModule-en:

    The module that only supports for single-step (``step_mode == 's'``)
    """
    def supported_step_mode(self):
        return ('s', )

class MultiStepModule(StepModule):
    """
    * :ref:`API in English <MultiStepModule-en>`

    .. _MultiStepModule-cn:

    只支持多步的模块 (``step_mode == 'm'``)。

    * :ref:`中文 API <MultiStepModule-cn>`

    .. _MultiStepModule-en:

    The module that only supports for multi-step (``step_mode == 'm'``)
    """
    def supported_step_mode(self):
        return ('m', )

